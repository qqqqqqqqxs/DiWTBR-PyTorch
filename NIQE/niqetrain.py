import numpy as np
import scipy.misc
import scipy.io
import os
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import numpy as np
import scipy.ndimage
import numpy as np
import scipy.special
import math
from scipy.ndimage import gaussian_filter
import glob

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    # Flatten imdata
    imdata.shape = (len(imdata.flat),)
    
    # Separate positive and negative data before squaring
    left_data = imdata[imdata < 0]
    right_data = imdata[imdata >= 0]
    
    # Now square the data
    left_data_squared = left_data * left_data
    right_data_squared = right_data * right_data
    
    # Calculate means of the squared data
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data_squared) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data_squared))
    if len(right_data_squared) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data_squared))

    # Calculate gamma_hat
    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf

    # Solve for r_hat norm
    imdata_squared = imdata * imdata  # Squared imdata
    imdata2_mean = np.mean(imdata_squared)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata_squared))
    else:
        r_hat = np.inf

    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # Solve for alpha by guessing values that minimize rhat_norm
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    # Calculate gamma values
    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    # Calculate AGGD ratio
    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # Calculate N
    N = (br - bl) * (gam2 / gam1)

    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho))
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    """
    Compute MSCN coefficients using Gaussian filtering with correlate1d.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        C (float): Stabilizing constant.
        avg_window (numpy.ndarray): Gaussian kernel for filtering.
        extend_mode (str): Padding mode for boundaries.

    Returns:
        tuple: MSCN coefficients, variance, and mean.
    """
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)

    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')

    # Apply Gaussian filtering along rows and columns
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)

    # Compute variance and MSCN coefficients
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    mscn_coeffs = (image - mu_image) / (var_image + C)

    return mscn_coeffs, var_image, mu_image

def apply_gaussian_filter(image, sigma=7/6):
    """
    Apply Gaussian filtering similar to MATLAB's imgaussfilt.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        numpy.ndarray: Filtered image.
    """
    return gaussian_filter(image, sigma=sigma, mode='nearest')


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    features = np.array([
        alpha_m, (bl + br) / 2.0,
        alpha1, N1, bl1, br1,  # (V)
        alpha2, N2, bl2, br2,  # (H)
        alpha3, N3, bl3, br3,  # (D1)
        alpha4, N4, bl4, br4   # (D2)
    ])
    return features

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)  # Fixed np.int deprecation
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    img2 = np.array(Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.BICUBIC), dtype=np.float32)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    return feats

def train_niqe_model(image_folders, output_mat_path):
    """
    参数说明：
    image_folders : list of str 
        包含两个文件夹路径的列表，例如 ["folder1", "folder2"]
    output_mat_path : str
        输出参数文件的路径
    """
    all_features = []

    # 遍历所有文件夹
    for folder in image_folders:
        # 支持多种图像格式，不区分大小写
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder, ext), recursive=True))
            image_paths.extend(glob.glob(os.path.join(folder, ext.upper()), recursive=True))

        # 处理每个图像
        for img_path in image_paths:
            try:
                # 读取并转换为灰度图
                img = np.array(Image.open(img_path).convert('L'))
                # 提取特征
                feats = get_patches_train_features(img, patch_size=96)
                all_features.append(feats)
                print(f"Processed: {img_path}")
            except Exception as e:
                print(f"Skipped {img_path} due to error: {str(e)}")
                continue

    # 合并特征并计算统计量
    all_features = np.vstack(all_features)
    pop_mu = np.mean(all_features, axis=0)
    pop_cov = np.cov(all_features, rowvar=False)

    # 保存参数文件
    scipy.io.savemat(output_mat_path, {
        'clean_mean': pop_mu,
        'clean_cov': pop_cov
    })
    print(f"Training completed. Parameters saved to {output_mat_path}")

# ---------------------- 执行训练 ----------------------
if __name__ == "__main__":
    # 配置两个训练图像文件夹路径
    train_folders = [
        "/home/jupyter-qxs/Mynet/bokeh",
        "/home/jupyter-qxs/EBBresult/GT"
        "/mnt/data0/qiuxs/ICME_dataset/defocus_train_test_choose/train/FNumber_2"
        "/mnt/data0/qiuxs/ICME_dataset/defocus_train_test_choose/validation/FNumber_2"
    ]
    
    # 启动训练
    train_niqe_model(
        image_folders=train_folders,
        output_mat_path="custom_niqe_params.mat"
    )