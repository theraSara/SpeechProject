import torch
import torch.nn.functional as F

# z_s: student embedding 
# z_t: teacher embedding 
# z_ts: list of teacher embedding 

# assign weight to each teacher based on how good its embedding space 
# good teacher: compact within class, far apart between class prototypes 
def compute_teacher_weights(z_ts, n_class, n_sample):
    scores = []
    for z_t in z_ts:
        z = z_t.view(n_class, n_sample, -1)
        protos = z.mean(1)
        # how spread out samples are around their class prototype (smaller is better)
        # average squared distance from each sample to its class prototype (average within class variance)
        intra = (z - protos.unsqueeze(1)).pow(2).sum(-1).mean() 
        # how far apart the class prototypes (larger is better)
        # pairwise Euclidean distances between all prototypes (average prototype separation)
        inter = torch.cdist(protos, protos).mean()  
        # teacher score = separation / compactness
        # trust teacher whose embeddings from tight class clusters and well-separated prototype
        scores.append(inter / (intra + 1e-8))
    return F.softmax(torch.stack(scores), dim=0)

# point0to-pont loss 
# matches each student embedding to the corresponding teacher embedding 
# transfrs raw feature information from teacher to student 
# limitation: too strict (wants exact embedding matching), even though for metric learning the exact coordinates may matter less than relationships 
def p2p_loss(z_s, z_t):
    return F.mse_loss(z_s, z_t)

# structure-to-structure loss 
# two components: global similarity matching + prototype-relative residual matching (no angle term here)
def s2s_loss(z_s, z_t, n_sample, n_class):
    B = z_s.size(0) # no examples in one episode: n_class * n_sample 
    
    # similarity matrix matching (geometry)
    # dot-product: preserve relationships among all samples (relative alignment, pairwaise affinity, cluster structure) 
    dist_s = torch.cdist(z_s, z_s, p=2)
    dist_t = torch.cdist(z_t, z_t, p=2)

    pair_mask = ~torch.eye(B, dtype=torch.bool, device=z_s.device)
    mu_s = dist_s[pair_mask].mean().clamp_min(1e-12)
    mu_t = dist_t[pair_mask].mean().clamp_min(1e-12)

    phi_s = dist_s / mu_s
    phi_t = dist_t / mu_t
    loss_dist = F.smooth_l1_loss(phi_s[pair_mask], phi_t[pair_mask]) # more stable than MSE
    
    # prototype-relative residual matching
    # preserve how each sample sits relative to its class center
    z_s_view = z_s.reshape(n_class, n_sample, -1)
    z_t_view = z_t.reshape(n_class, n_sample, -1)
    proto_s = z_s_view.mean(1)
    proto_t = z_t_view.mean(1)
    # for each sample, compute its residual vector relative to its own class prototype
    # if a sample is slightly above and right of its prototype in teacher space, the student should also place it similarly relative to the student prototype
    dist_proto_s = torch.cdist(z_s, proto_s, p=2) / mu_s
    dist_proto_t = torch.cdist(z_t, proto_t, p=2) / mu_t
    loss_proto = F.smooth_l1_loss(dist_proto_s, dist_proto_t)

    # angle-wise relation matching
    idx = torch.arange(B, device=z_s.device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing='ij')
    triplet_mask = (i != j) & (i != k) & (j != k)

    u_s_ij = z_s[i] - z_s[j]
    u_s_kj = z_s[k] - z_s[j]
    u_t_ij = z_t[i] - z_t[j]
    u_t_kj = z_t[k] - z_t[j]

    cos_s = F.cosine_similarity(u_s_ij, u_s_kj, dim=-1)
    cos_t = F.cosine_similarity(u_t_ij, u_t_kj, dim=-1)
    loss_angle = F.smooth_l1_loss(cos_s[triplet_mask], cos_t[triplet_mask])

    return loss_dist + loss_angle + loss_proto

# aligns teacher and student probability distributions 
def kl_loss(z_s, z_t, n_sample, n_class, temperature=1.0):
    z_s_view = z_s.view(n_class, n_sample, -1)
    z_t_view = z_t.view(n_class, n_sample, -1)

    proto_s = z_s_view.mean(1)
    proto_t = z_t_view.mean(1)

    rho_s = torch.cdist(z_s, proto_s).pow(2)
    rho_t = torch.cdist(z_t, proto_t).pow(2)

    log_p_s = F.log_softmax(-rho_s / temperature, dim=1)
    p_t = F.softmax(-rho_t / temperature, dim=1)

    return F.kl_div(log_p_s, p_t, reduction='batchmean')
