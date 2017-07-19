"""Pose models for visualization."""
# pylint: disable=invalid-name, no-member
import os.path as path
import cPickle as pickle
from collections import OrderedDict

def enum(*sequential):
    """Reversible, ordered enum."""
    kv_tuples_complete = zip(sequential, range(len(sequential)))
    enums = OrderedDict(kv_tuples_complete)
    reverse = OrderedDict((value, key) for key, value in enums.iteritems())
    enums['keys'] = enums.keys()[:]  # Ignore the reverse_mapping
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)


def named_enum(kv_tuples):
    """Reversible, ordered enum."""
    kv_tuples_complete = kv_tuples
    enums = OrderedDict(kv_tuples_complete)
    reverse = OrderedDict((value, key) for key, value in enums.iteritems())
    enums['keys'] = enums.keys()[:]
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

def get_pose_names(images):
    """Robust pose search."""
    pose_names = []
    for image_name in images:
        if path.exists(image_name + '_pose.npz'):
            pose_names.append(image_name + '_pose.npz')
            continue
        elif path.exists(image_name + '_joints.npz'):
            pose_names.append(image_name + '_joints.npz')
            continue
        elif path.exists(image_name + '_pose.npy'):
            pose_names.append(image_name + '_pose.npy')
            continue
        elif path.exists(image_name + '_joints.npy'):
            pose_names.append(image_name + '_joints.npy')
            continue
        try:
            image_idx = int(path.basename(image_name)[:5])
        except:  # pylint: disable=bare-except
            pose_names.append(image_name + '_pose.npz')
            continue
        if path.exists(path.join(path.dirname(image_name),
                                   '%05d_joints.npy' % (image_idx))):
            pose_names.append(path.join(path.dirname(image_name),
                                         '%05d_joints.npy' % (image_idx)))
            continue
        elif path.exists(path.join(path.dirname(image_name),
                                     '%05d_joints.npz' % (image_idx))):
            pose_names.append(path.join(path.dirname(image_name),
                                         '%05d_joints.npz' % (image_idx)))
            continue
        else:
            pose_names.append(image_name + '_pose.npz')
            continue
    return pose_names

joints_lsp = enum('rankle', 'rknee', 'rhip',
                  'lhip', 'lknee', 'lankle',
                  'rwrist', 'relbow', 'rshoulder',
                  'lshoulder', 'lelbow', 'lwrist',
                  'neck', 'head')


connections_lsp = [(joints_lsp.head, joints_lsp.neck, (0, 255, 0), False),
                   (joints_lsp.neck, joints_lsp.rshoulder, (0, 255, 0), False),
                   (joints_lsp.neck, joints_lsp.lshoulder, (0, 255, 0), True),
                   (joints_lsp.rshoulder, joints_lsp.relbow, (0, 0, 255), False),
                   (joints_lsp.relbow, joints_lsp.rwrist, (0, 0, 255), False),
                   (joints_lsp.lshoulder, joints_lsp.lelbow, (0, 0, 255), True),
                   (joints_lsp.lelbow, joints_lsp.lwrist, (0, 0, 255), True),
                   (joints_lsp.rshoulder, joints_lsp.rhip, (255, 0, 0), False),
                   (joints_lsp.rhip, joints_lsp.rknee, (255, 255, 0), False),
                   (joints_lsp.rknee, joints_lsp.rankle, (255, 255, 0), False),
                   (joints_lsp.lshoulder, joints_lsp.lhip, (255, 0, 0), True),
                   (joints_lsp.lhip, joints_lsp.lknee, (255, 255, 0), True),
                   (joints_lsp.lknee, joints_lsp.lankle, (255, 255, 0), True)]
lr_swap_lsp = {joints_lsp.head: joints_lsp.head,
               joints_lsp.neck: joints_lsp.neck,
               joints_lsp.rshoulder: joints_lsp.lshoulder,
               joints_lsp.lshoulder: joints_lsp.rshoulder,
               joints_lsp.relbow: joints_lsp.lelbow,
               joints_lsp.lelbow: joints_lsp.relbow,
               joints_lsp.rwrist: joints_lsp.lwrist,
               joints_lsp.lwrist: joints_lsp.rwrist,
               joints_lsp.rhip: joints_lsp.lhip,
               joints_lsp.lhip: joints_lsp.rhip,
               joints_lsp.rknee: joints_lsp.lknee,
               joints_lsp.lknee: joints_lsp.rknee,
               joints_lsp.rankle: joints_lsp.lankle,
               joints_lsp.lankle: joints_lsp.rankle}
# For direct access.
rlswap_lsp = [-1 for _ in range(14)]
for source_joint_idx, target_joint_idx in lr_swap_lsp.items():
    rlswap_lsp[target_joint_idx] = source_joint_idx
for val in rlswap_lsp:
    assert val > -1
rlswap_lsp = tuple(rlswap_lsp)


################################## Segmentation ###############################

# pylint: disable=bad-whitespace
regions = named_enum(OrderedDict([
    ('rhand',     (253, 107,   0)),  # 1
    ('rwrist',    (253,  90,  97)),  # 2
    ('rlarm',     (249, 178,   3)),  # 3
    ('relbow',    ( 79,  22,  30)),  # 4
    ('ruarm',     (255, 218,   0)),  # 5
    ('rshoulder', ( 84,  81,  17)),  # 6
    ('rubody',    (211, 228,  94)),  # 7
    ('rlbody',    (154, 165,  46)),  # 8
    ('ruleg',     (110,  43,  84)),  # 9
    ('rknee',     (66,   28,  43)),  # 10
    ('rlleg',     (165,  72,  26)),  # 11
    ('rankle',    (249, 111,  72)),  # 12
    ('rfoot',     (219, 155,  60)),  # 13
    ('lhand',     (82,  104, 125)),  # 14
    ('lwrist',    (98,  233, 115)),  # 15
    ('llarm',     (3,   165, 122)),  # 16
    ('lelbow',    (28,   28,  48)),  # 17
    ('luarm',     (38,  159,  36)),  # 18
    ('lshoulder', (35,   67,  48)),  # 19
    ('lubody',    (253, 228,  94)),  # 20
    ('llbody',    (181, 151, 103)),  # 21
    ('luleg',     (45,   49,  72)),  # 22
    ('lknee',     (84,   28,  37)),  # 23
    ('llleg',     (249, 153,  86)),  # 24
    ('lankle',    (102, 229,  66)),  # 25
    ('lfoot',     (248, 206,  74)),  # 26
    ('luhead',    (225,  22,  23)),  # 27
    ('ruhead',    (149,   1,  23)),  # 28
    ('rlhead',    (10,   77,  22)),  # 29
    ('llhead',    (38,   26,  24)),  # 30
    ('neck',      (167, 207,  29)),  # 31
    ('crotch',    (191, 175,  95))   # --
    # This is an artifact of coloring, since
    # parts of this regions could not be properly colored. It will
    # be mapped to right lower body.
]).items())
regions_preproject = named_enum(OrderedDict([
    ('rhand',     (254, 118,   0)),  # 1
    ('rwrist',    (254,  99, 195)),  # 2
    ('rlarm',     (250, 196,   7)),  # 3
    ('relbow',    ( 81,  25,  62)),  # 4
    ('ruarm',     (255, 241,   0)),  # 5
    ('rshoulder', ( 84,  90,  36)),  # 6
    ('rubody',    (211, 249, 190)),  # 7
    ('rlbody',    (155, 183,  93)),  # 8
    ('ruleg',     (111,  47, 168)),  # 9
    ('rknee',     (67,   32,  87)),  # 10
    ('rlleg',     (165,  81,  53)),  # 11
    ('rankle',    (249, 122, 144)),  # 12
    ('rfoot',     (219, 170, 122)),  # 13
    ('lhand',     (82,  115, 252)),  # 14
    ('lwrist',    (98,  255, 230)),  # 15
    ('llarm',     (4,   183, 244)),  # 16
    ('lelbow',    (29,   32,  96)),  # 17
    ('luarm',     (38,  174,  73)),  # 18
    ('lshoulder', (37,   74,  97)),  # 19
    ('lubody',    (254, 249, 190)),  # 20
    ('llbody',    (181, 166, 208)),  # 21
    ('luleg',     (46,   55, 145)),  # 22
    ('lknee',     (85,   31,  76)),  # 23
    ('llleg',     (249, 167, 173)),  # 24
    ('lankle',    (102, 251, 134)),  # 25
    ('lfoot',     (248, 225, 150)),  # 26
    ('luhead',    (225,  25,  48)),  # 27
    ('ruhead',    (151,   3,  48)),  # 28
    ('rlhead',    (12,   84,  47)),  # 29
    ('llhead',    (39,   31,  49)),  # 30
    ('neck',      (167, 226,  58)),  # 31
    ('crotch',    (191, 175,  95))   # --
    # This is an artifact of coloring, since
    # parts of this regions could not be properly colored. It will
    # be mapped to right lower body.
]).items())
# pylint: enable=bad-whitespace

six_region_groups = [
    [regions.ruhead, regions.rlhead, regions.luhead, regions.llhead],
    [regions.neck, regions.rubody, regions.rlbody, regions.lubody,
     regions.llbody, regions.rshoulder, regions.lshoulder],
    [regions.ruarm, regions.relbow, regions.rlarm, regions.rwrist, regions.rhand],
    [regions.luarm, regions.lelbow, regions.llarm, regions.lwrist, regions.lhand],
    [regions.ruleg, regions.rknee, regions.rlleg, regions.rankle, regions.rfoot],
    [regions.luleg, regions. lknee, regions.llleg, regions.lankle, regions.lfoot],
]

with open(path.join(path.dirname(__file__),
                    'landmarks.pkl'), 'rb') as inf:
    landmark_mesh_91 = pickle.load(inf)

landmarks_91 = enum(
    *(joints_lsp.reverse_mapping.values()[:-2] + landmark_mesh_91.keys()))

rlswap_landmarks_91 = []
for idx, name in landmarks_91.reverse_mapping.items():
    if name.startswith('l'):
        rlswap_landmarks_91.append(landmarks_91.__dict__['r' + name[1:]])
    elif name.startswith('r'):
        rlswap_landmarks_91.append(landmarks_91.__dict__['l' + name[1:]])
    else:
        rlswap_landmarks_91.append(idx)
rlswap_landmarks_91 = tuple(rlswap_landmarks_91)

reduction_91tolsp = tuple(range(12) + [landmarks_91.neck, landmarks_91.head_top])

THICKNESS_THIN = 2
THICKNESS_THICK = 4
connections_landmarks_91 = [
    # Head.
    (landmarks_91.head_top, landmarks_91.neck, regions.ruhead, False, THICKNESS_THICK),
    (landmarks_91.head_top, landmarks_91.head_back, regions.ruhead, False, THICKNESS_THIN),
    (landmarks_91.head_top, landmarks_91.nose, regions.ruhead, False, THICKNESS_THIN),
    (landmarks_91.nose, landmarks_91.throat, regions.rlhead, False, THICKNESS_THIN),
    (landmarks_91.throat, landmarks_91.neck, regions.rlhead, False, THICKNESS_THIN),
    (landmarks_91.neck, landmarks_91.head_back, regions.rlhead, False, THICKNESS_THIN),
    (landmarks_91.neck, landmarks_91.lear, regions.rlhead, False, THICKNESS_THIN),
    (landmarks_91.lear, landmarks_91.head_top, regions.luhead, False, THICKNESS_THIN),
    (landmarks_91.head_top, landmarks_91.rear, regions.ruhead, False, THICKNESS_THIN),
    (landmarks_91.rear, landmarks_91.neck, regions.rlhead, False, THICKNESS_THIN),

    # Shoulder area.
    (landmarks_91.neck, landmarks_91.rshoulder, regions.rubody, False, THICKNESS_THICK),
    (landmarks_91.neck, landmarks_91.rshoulder_top, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.neck, landmarks_91.rshoulder_back, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.neck, landmarks_91.rshoulder_front, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.rshoulder_top, landmarks_91.rshoulder, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.rshoulder_front, landmarks_91.rshoulder, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.rshoulder_back, landmarks_91.rshoulder, regions.rubody, False, THICKNESS_THIN),

    (landmarks_91.neck, landmarks_91.lshoulder, regions.lubody, False, THICKNESS_THICK),
    (landmarks_91.neck, landmarks_91.lshoulder_top, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.neck, landmarks_91.lshoulder_back, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.neck, landmarks_91.lshoulder_front, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.lshoulder_top, landmarks_91.lshoulder, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.lshoulder_front, landmarks_91.lshoulder, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.lshoulder_back, landmarks_91.lshoulder, regions.lubody, False, THICKNESS_THIN),

    # Core.
    (landmarks_91.neck, landmarks_91.lshoulder, regions.rubody, True, THICKNESS_THICK),
    (landmarks_91.lhip, landmarks_91.lshoulder, regions.lubody, True, THICKNESS_THICK),
    (landmarks_91.rhip, landmarks_91.rshoulder, regions.rubody, False, THICKNESS_THICK),
    (landmarks_91.rhip, landmarks_91.rknee, regions.ruleg, False, THICKNESS_THICK),
    (landmarks_91.lhip, landmarks_91.lknee, regions.luleg, True, THICKNESS_THICK),
    (landmarks_91.lknee, landmarks_91.lankle, regions.llleg, True, THICKNESS_THICK),
    (landmarks_91.rknee, landmarks_91.rankle, regions.rlleg, False, THICKNESS_THICK),
    (landmarks_91.lshoulder, landmarks_91.lelbow, regions.luarm, True, THICKNESS_THICK),
    (landmarks_91.lwrist, landmarks_91.lelbow, regions.llarm, True, THICKNESS_THICK),
    (landmarks_91.rshoulder, landmarks_91.relbow, regions.ruarm, False, THICKNESS_THICK),
    (landmarks_91.rwrist, landmarks_91.relbow, regions.rlarm, False, THICKNESS_THICK),

    # Right foot.
    (landmarks_91.rankle, landmarks_91.rheel, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rankle, landmarks_91.rankle_outer, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rankle, landmarks_91.rankle_inner, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rankle, landmarks_91.rbigtoe, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rbigtoe, landmarks_91.rankle_inner, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rbigtoe, landmarks_91.rankle_outer, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rankle_outer, landmarks_91.rheel, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rankle_inner, landmarks_91.rheel, regions.rfoot, False, THICKNESS_THIN),
    (landmarks_91.rbigtoe, landmarks_91.rheel, regions.rfoot, False, THICKNESS_THIN),
    # Left foot.
    (landmarks_91.lankle, landmarks_91.lheel, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lankle, landmarks_91.lankle_outer, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lankle, landmarks_91.lankle_inner, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lankle, landmarks_91.lbigtoe, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lbigtoe, landmarks_91.lankle_inner, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lbigtoe, landmarks_91.lankle_outer, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lankle_outer, landmarks_91.lheel, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lankle_inner, landmarks_91.lheel, regions.lfoot, False, THICKNESS_THIN),
    (landmarks_91.lbigtoe, landmarks_91.lheel, regions.lfoot, False, THICKNESS_THIN),
    # Right leg.
    (landmarks_91.rankle_outer, landmarks_91.rlleg_outer, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rankle_inner, landmarks_91.rlleg_inner, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rheel, landmarks_91.rlleg_back, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rankle, landmarks_91.rlleg_front, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_outer, landmarks_91.rlleg_outer, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_inner, landmarks_91.rlleg_inner, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_back, landmarks_91.rlleg_back, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_front, landmarks_91.rlleg_front, regions.rlleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_outer, landmarks_91.ruleg_outer, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_inner, landmarks_91.ruleg_inner, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_back, landmarks_91.ruleg_back, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.rknee_front, landmarks_91.ruleg_front, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.rhip, landmarks_91.ruleg_front, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.rhip, landmarks_91.ruleg_back, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.rhip, landmarks_91.ruleg_inner, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.ruleg_outer, landmarks_91.rhip, regions.ruleg, False, THICKNESS_THIN),
    (landmarks_91.ruleg_outer, landmarks_91.rhip_outer, regions.ruleg, False, THICKNESS_THIN),
    # Left leg.
    (landmarks_91.lankle_outer, landmarks_91.llleg_outer, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lankle_inner, landmarks_91.llleg_inner, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lheel, landmarks_91.llleg_back, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lankle, landmarks_91.llleg_front, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_outer, landmarks_91.llleg_outer, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_inner, landmarks_91.llleg_inner, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_back, landmarks_91.llleg_back, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_front, landmarks_91.llleg_front, regions.llleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_outer, landmarks_91.luleg_outer, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_inner, landmarks_91.luleg_inner, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_back, landmarks_91.luleg_back, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lknee_front, landmarks_91.luleg_front, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lhip, landmarks_91.luleg_front, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lhip, landmarks_91.luleg_back, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lhip, landmarks_91.luleg_inner, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.lhip, landmarks_91.luleg_outer, regions.luleg, False, THICKNESS_THIN),
    (landmarks_91.luleg_outer, landmarks_91.lhip_outer, regions.luleg, False, THICKNESS_THIN),

    # Body front region.
    (landmarks_91.rhip_outer, landmarks_91.rwaist, regions.rlbody, False, THICKNESS_THIN),
    (landmarks_91.lhip_outer, landmarks_91.lwaist, regions.llbody, False, THICKNESS_THIN),
    (landmarks_91.belly_button, landmarks_91.rwaist, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.belly_button, landmarks_91.lwaist, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.belly_button, landmarks_91.rpapilla, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.belly_button, landmarks_91.lpapilla, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.rshoulder_front, landmarks_91.rwaist, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.rshoulder_front, landmarks_91.rpapilla, regions.rubody, False, THICKNESS_THIN),
    (landmarks_91.lshoulder_front, landmarks_91.lwaist, regions.lubody, False, THICKNESS_THIN),
    (landmarks_91.lshoulder_front, landmarks_91.lpapilla, regions.lubody, False, THICKNESS_THIN),

    # Arms.
    (landmarks_91.rshoulder_front, landmarks_91.ruarm_inner, regions.ruarm, False, THICKNESS_THIN),
    (landmarks_91.rshoulder, landmarks_91.ruarm_outer, regions.ruarm, False, THICKNESS_THIN),
    (landmarks_91.relbow_inner, landmarks_91.ruarm_inner, regions.ruarm, False, THICKNESS_THIN),
    (landmarks_91.relbow_outer, landmarks_91.ruarm_outer, regions.ruarm, False, THICKNESS_THIN),
    (landmarks_91.relbow_inner, landmarks_91.rlarm_lower, regions.rlarm, False, THICKNESS_THIN),
    (landmarks_91.relbow_outer, landmarks_91.rlarm_upper, regions.rlarm, False, THICKNESS_THIN),
    (landmarks_91.rwrist, landmarks_91.rlarm_lower, regions.rlarm, False, THICKNESS_THIN),
    (landmarks_91.rwrist, landmarks_91.rlarm_upper, regions.rlarm, False, THICKNESS_THIN),

    (landmarks_91.lshoulder_front, landmarks_91.luarm_inner, regions.luarm, False, THICKNESS_THIN),
    (landmarks_91.lshoulder, landmarks_91.luarm_outer, regions.luarm, False, THICKNESS_THIN),
    (landmarks_91.lelbow_inner, landmarks_91.luarm_inner, regions.luarm, False, THICKNESS_THIN),
    (landmarks_91.lelbow_outer, landmarks_91.luarm_outer, regions.luarm, False, THICKNESS_THIN),
    (landmarks_91.lelbow_inner, landmarks_91.llarm_lower, regions.llarm, False, THICKNESS_THIN),
    (landmarks_91.lelbow_outer, landmarks_91.llarm_upper, regions.llarm, False, THICKNESS_THIN),
    (landmarks_91.lwrist, landmarks_91.llarm_lower, regions.llarm, False, THICKNESS_THIN),
    (landmarks_91.lwrist, landmarks_91.llarm_upper, regions.llarm, False, THICKNESS_THIN),
]

lm_region_mapping = {
    landmarks_91.lankle_outer: regions.lankle,
    landmarks_91.lankle_inner: regions.lankle,
    landmarks_91.lheel: regions.lfoot,
    landmarks_91.lbigtoe: regions.lfoot,
    landmarks_91.lankle: regions.lankle,

    landmarks_91.llleg_outer: regions.llleg,
    landmarks_91.llleg_inner: regions.llleg,
    landmarks_91.llleg_front: regions.llleg,
    landmarks_91.llleg_back: regions.llleg,

    landmarks_91.lknee: regions.lknee,
    landmarks_91.lknee_front: regions.lknee,
    landmarks_91.lknee_back: regions.lknee,
    landmarks_91.lknee_outer: regions.lknee,
    landmarks_91.lknee_inner: regions.lknee,

    landmarks_91.luleg_front: regions.luleg,
    landmarks_91.luleg_back: regions.luleg,
    landmarks_91.luleg_inner: regions.luleg,
    landmarks_91.luleg_outer: regions.luleg,

    landmarks_91.rankle_outer: regions.rankle,
    landmarks_91.rankle_inner: regions.rankle,
    landmarks_91.rheel: regions.rfoot,
    landmarks_91.rbigtoe: regions.rfoot,
    landmarks_91.rankle: regions.rankle,

    landmarks_91.rlleg_outer: regions.rlleg,
    landmarks_91.rlleg_inner: regions.rlleg,
    landmarks_91.rlleg_front: regions.rlleg,
    landmarks_91.rlleg_back: regions.rlleg,

    landmarks_91.rknee: regions.rknee,
    landmarks_91.rknee_front: regions.rknee,
    landmarks_91.rknee_back: regions.rknee,
    landmarks_91.rknee_outer: regions.rknee,
    landmarks_91.rknee_inner: regions.rknee,

    landmarks_91.ruleg_front: regions.ruleg,
    landmarks_91.ruleg_back: regions.ruleg,
    landmarks_91.ruleg_inner: regions.ruleg,
    landmarks_91.ruleg_outer: regions.ruleg,

    landmarks_91.lhip: regions.llbody,
    landmarks_91.lhip_outer: regions.llbody,
    landmarks_91.rhip: regions.rlbody,
    landmarks_91.rhip_outer: regions.rlbody,
    landmarks_91.lwaist: regions.llbody,
    landmarks_91.rwaist: regions.rlbody,
    landmarks_91.lshoulder: regions.lubody,
    landmarks_91.rshoulder: regions.rubody,
    landmarks_91.belly_button: regions.rlbody,
    landmarks_91.lpapilla: regions.lubody,
    landmarks_91.rpapilla: regions.rubody,

    landmarks_91.lwrist: regions.lwrist,
    landmarks_91.rwrist: regions.rwrist,
    landmarks_91.llarm_upper: regions.llarm,
    landmarks_91.llarm_lower: regions.llarm,
    landmarks_91.rlarm_upper: regions.rlarm,
    landmarks_91.rlarm_lower: regions.rlarm,

    landmarks_91.lelbow: regions.lelbow,
    landmarks_91.relbow: regions.relbow,
    landmarks_91.relbow_outer: regions.relbow,
    landmarks_91.relbow_inner: regions.relbow,
    landmarks_91.lelbow_outer: regions.lelbow,
    landmarks_91.lelbow_inner: regions.lelbow,

    landmarks_91.ruarm_inner: regions.ruarm,
    landmarks_91.ruarm_outer: regions.ruarm,
    landmarks_91.luarm_inner: regions.luarm,
    landmarks_91.luarm_outer: regions.luarm,

    landmarks_91.rshoulder_front: regions.rubody,
    landmarks_91.rshoulder_top: regions.rubody,
    landmarks_91.rshoulder_back: regions.rubody,

    landmarks_91.lshoulder_front: regions.lubody,
    landmarks_91.lshoulder_top: regions.lubody,
    landmarks_91.lshoulder_back: regions.lubody,

    landmarks_91.neck: regions.neck,
    landmarks_91.head_top: regions.ruhead,
    landmarks_91.rear: regions.ruhead,
    landmarks_91.lear: regions.luhead,
    landmarks_91.nose: regions.ruhead,
    landmarks_91.throat: regions.neck,
    landmarks_91.head_back: regions.ruhead,
}

THICKNESS_THIN = 4
THICKNESS_THICK = 7
dots_landmarks_91 = {
    # Head.
    landmarks_91.head_top: ((0, 255, 0), THICKNESS_THICK),
    landmarks_91.neck: ((0, 255, 0), THICKNESS_THICK),
#    landmarks_91.reye: ((255, 255, 255), THICKNESS_THICK - 1),
#    landmarks_91.leye: ((255, 255, 255), THICKNESS_THICK - 1),
    landmarks_91.head_back: ((255, 255, 255), THICKNESS_THIN),
    landmarks_91.throat: ((255, 255, 255), THICKNESS_THIN),
    landmarks_91.lear: ((255, 255, 255), THICKNESS_THIN),
    landmarks_91.rear: ((255, 255, 255), THICKNESS_THIN),
    #landmarks_91.rmcorner: ((255, 255, 255), THICKNESS_THIN),
    #landmarks_91.lmcorner: ((255, 255, 255), THICKNESS_THIN),

    # Shoulder area.
    landmarks_91.rshoulder: ((255, 0, 0), THICKNESS_THICK),
    landmarks_91.lshoulder: ((255, 0, 0), THICKNESS_THICK),
    landmarks_91.rshoulder_top: ((255, 0, 0), THICKNESS_THIN),
    landmarks_91.rshoulder_back: ((255, 0, 0), THICKNESS_THIN),
    landmarks_91.rshoulder_front: ((255, 0, 0), THICKNESS_THIN),

    landmarks_91.rhip: ((255, 0, 0), THICKNESS_THICK),
    landmarks_91.lhip: ((255, 0, 0), THICKNESS_THICK),
    landmarks_91.rknee: ((255, 255, 0), THICKNESS_THICK),
    landmarks_91.lknee: ((255, 255, 0), THICKNESS_THICK),
    landmarks_91.relbow: ((0, 0, 255), THICKNESS_THICK),
    landmarks_91.lelbow: ((0, 0, 255), THICKNESS_THICK),
    landmarks_91.rwrist: ((0, 0, 255), THICKNESS_THICK),
    landmarks_91.lwrist: ((0, 0, 255), THICKNESS_THICK),
    landmarks_91.rankle: ((255, 255, 0), THICKNESS_THICK),
    landmarks_91.lankle: ((255, 255, 0), THICKNESS_THICK),

    landmarks_91.rheel: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.lheel: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.rankle_outer: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.rankle_inner: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.lankle_outer: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.lankle_inner: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.lbigtoe: ((255, 255, 0), THICKNESS_THIN),
    landmarks_91.rbigtoe: ((255, 255, 0), THICKNESS_THIN),

    landmarks_91.rpapilla: ((255, 0, 0), THICKNESS_THICK - 1),
    landmarks_91.lpapilla: ((255, 0, 0), THICKNESS_THICK - 1),
    landmarks_91.belly_button: ((255, 0, 0), THICKNESS_THICK - 1),
    }


lm_region_mapping = {
    landmarks_91.lankle_outer: regions.lankle,
    landmarks_91.lankle_inner: regions.lankle,
    landmarks_91.lheel: regions.lfoot,
    landmarks_91.lbigtoe: regions.lfoot,
    landmarks_91.lankle: regions.lankle,

    landmarks_91.llleg_outer: regions.llleg,
    landmarks_91.llleg_inner: regions.llleg,
    landmarks_91.llleg_front: regions.llleg,
    landmarks_91.llleg_back: regions.llleg,

    landmarks_91.lknee: regions.lknee,
    landmarks_91.lknee_front: regions.lknee,
    landmarks_91.lknee_back: regions.lknee,
    landmarks_91.lknee_outer: regions.lknee,
    landmarks_91.lknee_inner: regions.lknee,

    landmarks_91.luleg_front: regions.luleg,
    landmarks_91.luleg_back: regions.luleg,
    landmarks_91.luleg_inner: regions.luleg,
    landmarks_91.luleg_outer: regions.luleg,

    landmarks_91.rankle_outer: regions.rankle,
    landmarks_91.rankle_inner: regions.rankle,
    landmarks_91.rheel: regions.rfoot,
    landmarks_91.rbigtoe: regions.rfoot,
    landmarks_91.rankle: regions.rankle,

    landmarks_91.rlleg_outer: regions.rlleg,
    landmarks_91.rlleg_inner: regions.rlleg,
    landmarks_91.rlleg_front: regions.rlleg,
    landmarks_91.rlleg_back: regions.rlleg,

    landmarks_91.rknee: regions.rknee,
    landmarks_91.rknee_front: regions.rknee,
    landmarks_91.rknee_back: regions.rknee,
    landmarks_91.rknee_outer: regions.rknee,
    landmarks_91.rknee_inner: regions.rknee,

    landmarks_91.ruleg_front: regions.ruleg,
    landmarks_91.ruleg_back: regions.ruleg,
    landmarks_91.ruleg_inner: regions.ruleg,
    landmarks_91.ruleg_outer: regions.ruleg,

    landmarks_91.lhip: regions.llbody,
    landmarks_91.lhip_outer: regions.llbody,
    landmarks_91.rhip: regions.rlbody,
    landmarks_91.rhip_outer: regions.rlbody,
    landmarks_91.lwaist: regions.llbody,
    landmarks_91.rwaist: regions.rlbody,
    landmarks_91.lshoulder: regions.lubody,
    landmarks_91.rshoulder: regions.rubody,
    landmarks_91.belly_button: regions.rlbody,
    landmarks_91.lpapilla: regions.lubody,
    landmarks_91.rpapilla: regions.rubody,

    landmarks_91.lwrist: regions.lwrist,
    landmarks_91.rwrist: regions.rwrist,
    landmarks_91.llarm_upper: regions.llarm,
    landmarks_91.llarm_lower: regions.llarm,
    landmarks_91.rlarm_upper: regions.rlarm,
    landmarks_91.rlarm_lower: regions.rlarm,

    landmarks_91.lelbow: regions.lelbow,
    landmarks_91.relbow: regions.relbow,
    landmarks_91.relbow_outer: regions.relbow,
    landmarks_91.relbow_inner: regions.relbow,
    landmarks_91.lelbow_outer: regions.lelbow,
    landmarks_91.lelbow_inner: regions.lelbow,

    landmarks_91.ruarm_inner: regions.ruarm,
    landmarks_91.ruarm_outer: regions.ruarm,
    landmarks_91.luarm_inner: regions.luarm,
    landmarks_91.luarm_outer: regions.luarm,

    landmarks_91.rshoulder_front: regions.rubody,
    landmarks_91.rshoulder_top: regions.rubody,
    landmarks_91.rshoulder_back: regions.rubody,

    landmarks_91.lshoulder_front: regions.lubody,
    landmarks_91.lshoulder_top: regions.lubody,
    landmarks_91.lshoulder_back: regions.lubody,

    landmarks_91.neck: regions.neck,
    landmarks_91.head_top: regions.ruhead,
    landmarks_91.rear: regions.ruhead,
    landmarks_91.lear: regions.luhead,
    landmarks_91.nose: regions.ruhead,
    landmarks_91.throat: regions.neck,
    landmarks_91.head_back: regions.ruhead,
}
