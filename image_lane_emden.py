import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ============================================================
# 1. Load image and add Gaussian noise
# ============================================================

img = Image.open("lenna.png").convert("RGB")
img_array = np.array(img, dtype=np.float64)
img_norm = img_array / 255.0

height, width, channels = img_norm.shape

# Gaussian noise parameters
noise_sigma = 0.09   # standard deviation of noise
np.random.seed(0)

noise = noise_sigma * np.random.randn(height, width, channels)
noisy_img = np.clip(img_norm + noise, 0, 1)

# ============================================================
# 2. Initialize u and v with the noisy image
# ============================================================

u = noisy_img.copy()
v = noisy_img.copy()

# ============================================================
# 3. Parameters
# ============================================================

alpha = 0.45
beta  = 0.35
max_iter = 150
tol = 1e-4

omega = 0.78  # relaxation
lambda_data = 0.12
lambda_smooth = 1.0 - lambda_data

nl_max_u = 0.85
nl_max_v = 0.85

# Edge-preserving diffusion parameter
kappa = 0.14

# Early stopping based on SSIM
check_every = 2
patience = 3
ssim_tolerance = 1e-4

best_ssim = -np.inf
best_iter = 0
best_u = u.copy()
worse_count = 0

# ============================================================
# 4. Compute p(x,y) = 1/(1+R^2)
# ============================================================

x_center = width / 2.0
y_center = height / 2.0
X, Y = np.meshgrid(np.arange(width), np.arange(height))
R = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

p = 1.0 / (1.0 + R**2)
q = p.copy()

# ============================================================
# 5. Iterative Lane–Emden–Fowler + Edge-preserving diffusion
# ============================================================

for iteration in range(max_iter):

    u_old = u.copy()
    v_old = v.copy()

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for c in range(channels):

                # Nonlinear Lane–Emden terms
                nl_u = p[i, j] * (v_old[i, j, c] ** alpha)
                nl_v = q[i, j] * (u_old[i, j, c] ** beta)

                nl_u = np.clip(nl_u, 0.0, nl_max_u)
                nl_v = np.clip(nl_v, 0.0, nl_max_v)

                # Gradient for edge-preserving conductance
                gx = 0.5 * (u_old[i+1, j, c] - u_old[i-1, j, c])
                gy = 0.5 * (u_old[i, j+1, c] - u_old[i, j-1, c])
                grad_sq = gx * gx + gy * gy

                c_edge = np.exp(-grad_sq / (kappa * kappa))

                # Smoothing candidate
                u_smooth = 0.25 * (
                    u_old[i+1, j, c] + u_old[i-1, j, c] +
                    u_old[i, j+1, c] + u_old[i, j-1, c] +
                    nl_u
                )

                v_smooth = 0.25 * (
                    v_old[i+1, j, c] + v_old[i-1, j, c] +
                    v_old[i, j+1, c] + v_old[i, j-1, c] +
                    nl_v
                )

                # Edge-preserving blending
                u_candidate = c_edge * u_smooth + (1.0 - c_edge) * u_old[i, j, c]
                v_candidate = c_edge * v_smooth + (1.0 - c_edge) * v_old[i, j, c]

                # Semi-implicit update
                u[i, j, c] = (1 - omega) * u_old[i, j, c] + \
                              omega * (lambda_smooth * u_candidate +
                                       lambda_data  * noisy_img[i, j, c])

                v[i, j, c] = (1 - omega) * v_old[i, j, c] + \
                              omega * (lambda_smooth * v_candidate +
                                       lambda_data  * noisy_img[i, j, c])

    # Error check
    err = max(np.max(np.abs(u - u_old)), np.max(np.abs(v - v_old)))
    print(f"Iteration {iteration+1}: error = {err:.2e}")

    if err < tol:
        print("Convergence achieved based on error tolerance!")
        break

    # SSIM-based early stopping
    if (iteration + 1) % check_every == 0:
        current_ssim = ssim(img_norm, np.clip(u, 0, 1),
                            channel_axis=2, data_range=1.0)
        print(f"  -> SSIM at iteration {iteration+1}: {current_ssim:.4f}")

        if current_ssim > best_ssim + 1e-5:
            best_ssim = current_ssim
            best_iter = iteration + 1
            best_u = u.copy()
            worse_count = 0
        else:
            if current_ssim < best_ssim - ssim_tolerance:
                worse_count += 1
            else:
                worse_count = 0

        if worse_count >= patience:
            print(f"Early stopping: SSIM started to degrade (best at iter {best_iter}).")
            break

# Final restored image
restored = best_u

# ============================================================
# 6. Compute metrics
# ============================================================

mse_noisy = np.mean((img_norm - noisy_img) ** 2)
psnr_noisy = psnr(img_norm, noisy_img, data_range=1.0)
ssim_noisy = ssim(img_norm, noisy_img, channel_axis=2, data_range=1.0)

mse_rest = np.mean((img_norm - restored) ** 2)
psnr_rest = psnr(img_norm, restored, data_range=1.0)
ssim_rest = ssim(img_norm, restored, channel_axis=2, data_range=1.0)

print("\n=== Noisy Image Metrics ===")
print(f"MSE  = {mse_noisy:.6f}")
print(f"PSNR = {psnr_noisy:.4f} dB")
print(f"SSIM = {ssim_noisy:.4f}")

print("\n=== Restored Image Metrics ===")
print(f"MSE  = {mse_rest:.6f}")
print(f"PSNR = {psnr_rest:.4f} dB")
print(f"SSIM = {ssim_rest:.4f}")

# ============================================================
# 7. Show results
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_norm)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_img)
plt.title("Noisy Image (Gaussian)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.clip(restored, 0, 1))
plt.title(f"Restored Image (best iter {best_iter})")
plt.axis("off")

plt.tight_layout()
plt.show()
