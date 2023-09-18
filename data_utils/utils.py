import numpy as np
import random
import cv2
from PIL import Image

def extract_bounding_box(fg_img, fg_mask):
    ret, bin_fg_mask = cv2.threshold((fg_mask*255).astype(np.uint8), 127, 1, cv2.THRESH_BINARY)
    points = cv2.boundingRect(bin_fg_mask)

    bb_fg_mask = fg_mask[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]
    bb_fg_img = fg_img[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]

    return bb_fg_img, bb_fg_mask


def PIL_resize_with_antialiasing(img, shape):
    img[img<0] = 0
    img[img>1] = 1
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)/255.0

    return img


def do_composition(bb_fg_img, bb_fg_mask, bg_img, placement_config):
    """
    bb_fg_img: [0, 1.0]
    bb_fg_mask: [0, 1.0]
    bg_img: [0, 1.0]
    """
    OBJ_ATTRIBUTES = {"height":1.8}
    CAM_ATTRIBUTES = {"height":1.4, "HFOV":67.5/180*np.pi}

    # get ratio
    Pt_pixel_height = bg_img.shape[0] - placement_config['pt_h']
    tan_phi = (bg_img.shape[0]/2-Pt_pixel_height) / (bg_img.shape[0]*0.5) * np.tan(CAM_ATTRIBUTES["HFOV"]/2)
    OBJ_pixel_height = (tan_phi*np.tan(np.pi/2-CAM_ATTRIBUTES["HFOV"]/2)) / (1 - tan_phi*np.tan(np.pi/2-CAM_ATTRIBUTES["HFOV"]/2))  * OBJ_ATTRIBUTES['height'] / CAM_ATTRIBUTES['height'] * Pt_pixel_height
    RATIO = OBJ_pixel_height / bb_fg_mask.shape[0]

    if RATIO <= 0:
        RATIO = 0.5
    bb_h, bb_w = bb_fg_mask.shape
    nh, nw = int(RATIO * bb_h), int(RATIO * bb_w)
    # reshaped_fg_img = cv2.resize(bb_fg_img, (nw, nh), cv2.INTER_AREA)#cv2.INTER_CUBIC)
    # reshaped_bb_fg_mask = cv2.resize(bb_fg_mask, (nw, nh), cv2.INTER_AREA)#cv2.INTER_CUBIC)
    reshaped_fg_img = PIL_resize_with_antialiasing(bb_fg_img, (nw, nh))
    reshaped_bb_fg_mask = PIL_resize_with_antialiasing(bb_fg_mask, (nw, nh))
    reshaped_bb_fg_mask = np.expand_dims(reshaped_bb_fg_mask, 2)

    h_index, w_index = placement_config['pt_h'] - nh, placement_config['pt_w'] - nw//2
    # checking boundary
    if h_index < 0:
        h_index = 0
    elif h_index > (bg_img.shape[0]-nh):
        h_index = bg_img.shape[0]-nh
    if w_index < 0:
        w_index = 0
    elif w_index > (bg_img.shape[1]-nw):
        w_index = bg_img.shape[1]-nw
    
    new_fg_mask = np.zeros_like(bg_img)
    new_fg_mask[h_index:h_index + nh, w_index:w_index + nw] = reshaped_bb_fg_mask
    new_fg_img = np.zeros_like(bg_img)
    new_fg_img[h_index:h_index + nh, w_index:w_index + nw] = reshaped_fg_img

    # new_fg_mask = new_fg_mask / 255.0
    composited = new_fg_mask * new_fg_img + (1 - new_fg_mask) * bg_img

    # NORMALIZE
    composited[composited>1.0] = 1.0
    composited[composited<0.0] = 0.0
    # composited = composited / 255.0

    return composited.astype(np.float32), new_fg_mask.astype(np.float32)


def randomly_choosing_a_point(mask_placement):
    mask_placement[mask_placement>0.5] = 1.0
    mask_placement[mask_placement<=0.5] = 0.0

    indices = np.where(mask_placement == 1.0)
    indices = list(zip(indices[0], indices[1]))
    h, w = mask_placement.shape[0], mask_placement.shape[1]

    indice = indices[random.randint(0, len(indices) - 1)]
    # indice = indices[torch.randint(0, len(indices) - 1, (1,))]

    T = 400
    if indice[0] < T: # avoiding fg obj too small
        pt_h = T - indice[0] + T
        if pt_h > h:
            pt_h = h - 10
    else:
        pt_h = indice[0]

    return {"pt_h":pt_h, "pt_w":indice[1]}
