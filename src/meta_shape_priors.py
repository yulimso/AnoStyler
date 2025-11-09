import cv2
import random
import numpy as np
from scipy.ndimage import gaussian_filter1d, label


def msp_line(W=256, H=256):

    M = np.zeros((H, W), dtype=np.uint8)
    c = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    angle = np.random.uniform(0, 180)
    theta = np.deg2rad(angle)
    l = np.random.uniform(60, 200)
    s = np.random.randint(20, 40)
    x = np.linspace(-l/2, l/2, s)
    y = np.zeros_like(x)

    if np.random.rand() < 0.5:
        eps = np.random.normal(0, 14, size=s).astype(np.float32).reshape(-1, 1)
        y += cv2.GaussianBlur(eps, (1, 5), 2).flatten()

    min_thickness, max_thickness = 1, 7
    thickness_curve = (
        np.linspace(min_thickness, max_thickness, s // 2).tolist()
        + np.linspace(max_thickness, min_thickness, s - s // 2).tolist()
    )
    thickness_noise = np.random.uniform(
        -0.1 * (max_thickness - min_thickness),
        0.1 * (max_thickness - min_thickness),
        size=s
    )
    thickness_curve = np.array(thickness_curve) + thickness_noise
    thickness_curve = np.clip(thickness_curve, min_thickness, max_thickness)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    for i in range(s - 1):
        pt1 = R @ np.array([x[i], y[i]]) + c
        pt2 = R @ np.array([x[i+1], y[i+1]]) + c

        cv2.line(
            M,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            255,
            int(thickness_curve[i]),
            lineType=cv2.LINE_AA
        )

    _, M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)

    return M.astype(np.uint8)



def msp_dot(W=256, H=256):

    M = np.zeros((H, W), dtype=np.uint8)
    c = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    r = np.random.uniform(5, 35)
    s = np.random.randint(12, 30)
    theta = np.sort(np.random.uniform(0, 2*np.pi, s))
    alpha = np.random.uniform(0.6, 1.4)
    beta = np.random.uniform(0.05, 0.35)
    u = np.random.uniform(0, 1)

    if u >= 0.66:
        r_i = np.random.uniform(-beta*r, beta*r, s)
    else:
        r_i = np.random.normal(0, beta*r, s)

    x = c[0] + (r + r_i) * np.cos(theta) * alpha
    y = c[1] + (r + r_i) * np.sin(theta)

    contour = np.stack((x, y), axis=1).astype(np.int32)
    cv2.fillPoly(M, [contour], 255)

    if np.random.rand() < 0.5:
        k = np.random.choice([3, 5, 7])
        M = cv2.GaussianBlur(M, (k, k), 0)

    _, M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)

    return M.astype(np.uint8)



def msp_freeform(W=256, H=256):

    M = np.zeros((H, W), dtype=np.uint8)
    n_step = np.random.randint(300, 18001)
    sigma = np.random.uniform(2, 12)
    x, y = np.random.randint(0, W), np.random.randint(0, H)

    for _ in range(n_step):
        M[y, x] = 1
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        x = np.clip(x + dx, 0, W - 1)
        y = np.clip(y + dy, 0, H - 1)

    ksize = int(2 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    M = cv2.GaussianBlur(M.astype(np.float32), (ksize, ksize), sigmaX=sigma)

    if np.random.rand() < 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        M = cv2.dilate(M, kernel, 1)
        M = cv2.erode(M, kernel, 1)

    _, M = cv2.threshold(M, 0.5, 1, cv2.THRESH_BINARY)

    labeled, num = label(M, structure=np.ones((3, 3)))
    if num > 1:
        areas = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = 1 + np.argmax(areas)
        M = (labeled == largest).astype(np.uint8)

    return M.astype(np.uint8) * 255



def generate_meta_mask(W=256, H=256, m_max=5, alpha=0.7):

    base_W, base_H = 256, 256

    indices = np.arange(1, m_max + 1)
    logits = np.exp(-alpha * indices)
    probs = logits / logits.sum()

    m = np.random.choice(indices, p=probs)

    mask_final = np.zeros((base_H, base_W), dtype=np.uint8)
    shape_fns = [msp_line, msp_dot, msp_freeform]

    for _ in range(m):
        fn = random.choice(shape_fns)
        mask_i = fn(W=base_W, H=base_H)
        mask_final = np.clip(mask_final + (mask_i > 0).astype(np.uint8), 0, 1)

    mask_final = (mask_final * 255).astype(np.uint8)
    resized_mask = cv2.resize(mask_final, (W, H), interpolation=cv2.INTER_NEAREST)
    _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    return resized_mask