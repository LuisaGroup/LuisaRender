import numpy as np
from tqdm import tqdm


# static inline float dielectricReflectance(float eta, float cosThetaI)
# {
#     if (cosThetaI < 0.0f) {
#         eta = 1.0f/eta;
#         cosThetaI = -cosThetaI;
#     }
#     float cosThetaT;
#     float sinThetaTSq = eta*eta*(1.0f - cosThetaI*cosThetaI);
#     if (sinThetaTSq > 1.0f) {
#         cosThetaT = 0.0f;
#         return 1.0f;
#     }
#     cosThetaT = std::sqrt(max(1.0f - sinThetaTSq, 0.0f));
#
#     float Rs = (eta*cosThetaI - cosThetaT)/(eta*cosThetaI + cosThetaT);
#     float Rp = (eta*cosThetaT - cosThetaI)/(eta*cosThetaT + cosThetaI);
#
#     return (Rs*Rs + Rp*Rp)*0.5f;
# }

# static inline float computeDiffuseFresnel(float ior, const int sampleCount)
# {
#     double diffuseFresnel = 0.0;
#     float fb = Fresnel::dielectricReflectance(ior, 0.0f);
#     for (int i = 1; i <= sampleCount; ++i) {
#         float cosThetaSq = float(i)/sampleCount;
#         float fa = Fresnel::dielectricReflectance(ior, min(std::sqrt(cosThetaSq), 1.0f));
#         diffuseFresnel += double(fa + fb)*(0.5/sampleCount);
#         fb = fa;
#     }
#
#     return float(diffuseFresnel);
# }


def FrDielectric(eta: np.array, cosThetaI: float):
    eta = np.where(cosThetaI < 0, 1 / eta, eta)
    cosThetaI = np.abs(cosThetaI)
    sinThetaTSq = eta * eta * (1 - cosThetaI * cosThetaI)
    cosThetaT = np.sqrt(np.maximum(1 - sinThetaTSq, 0))
    Rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT)
    Rp = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI)
    F = 0.5 * (Rs * Rs + Rp * Rp)
    return np.where(sinThetaTSq >= 1, 1, F)


# computes integral[FrDielectric(eta, cos(theta)) * cos(theta)]
def FrDiffuse(eta: np.array, sampleCount: int):
    F = np.zeros_like(eta)
    fb = FrDielectric(eta, 0.)
    for i in tqdm(range(1, sampleCount + 1)):
        cosThetaSq = i / sampleCount
        fa = FrDielectric(eta, np.sqrt(cosThetaSq))
        F += (fa + fb) * (0.5 / sampleCount)
        fb = fa
    return F


def fit_small(eta: np.array, F: np.array):
    A = np.vstack([np.ones_like(eta), eta, eta ** 2, eta ** 3]).T
    c = np.linalg.lstsq(A, F, rcond=None)[0]
    print(c)
    return A @ c


def fit(eta: np.array, F: np.array):
    A = np.vstack([np.ones_like(eta), eta ** -1, eta ** -2]).T
    c = np.linalg.lstsq(A, F, rcond=None)[0]
    print(c)
    return A @ c


from matplotlib import pyplot as plt

if __name__ == "__main__":
    eta_small = np.arange(0.25, 1.00, 0.01)
    eta = np.arange(1.00, 4.00, 0.01)
    print(len(eta_small))
    print(len(eta))
    N = 100000
    F_small = FrDiffuse(eta_small, N)
    F = FrDiffuse(eta, N)
    F_small_recon = fit_small(eta_small, F_small)
    F_recon = fit(eta, F)
    print(F_small_recon)
    print(F_small - F_small_recon)
    print(np.max(np.abs(F_small - F_small_recon)))
    print(F_recon)
    print(F - F_recon)
    print(np.max(np.abs(F - F_recon)))
    plt.plot(eta_small, F_small, label="F small")
    plt.plot(eta, F, label="F")
    plt.plot(eta, F_recon, label="F recon", alpha=0.5, linestyle="--")
    plt.plot(eta_small, F_small_recon, label="F small recon", alpha=0.5, linestyle="--")
    plt.legend()

    plt.show()
