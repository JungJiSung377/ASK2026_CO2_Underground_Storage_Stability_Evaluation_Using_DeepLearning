import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

# [1] 리커 웨이브렛 생성 함수
def generate_ricker_wavelet(freq, dt, length):
    t = np.arange(-length/2, length/2, dt)
    pi2_f2_t2 = (np.pi**2) * (freq**2) * (t**2)
    wavelet = (1 - 2 * pi2_f2_t2) * np.exp(-pi2_f2_t2)
    return wavelet

# [2] 3D 지질 모델(공극률 + 투과율 + 지진파) 생성 함수
def create_geological_model(size=(64, 64, 64), sigma=1.0):
    raw_noise = np.random.rand(*size)
    porosity = gaussian_filter(raw_noise, sigma=sigma)

    # 범위를 [5%, 30%]로 선형 변환
    porosity = (porosity - np.mean(porosity)) * 2.5 + np.mean(porosity)
    porosity = (porosity - porosity.min()) / (porosity.max() - porosity.min())
    porosity = porosity * 0.25 + 0.05

    # Kozeny-Carman 관계식을 이용한 투과율 도출
    A = 1000 
    permeability = A * (porosity**3) / ((1 - porosity)**2)

    log_k = np.log10(permeability)
    perm_norm = (log_k - log_k.min()) / (log_k.max() - log_k.min())

    reflection_coeff = np.gradient(porosity, axis=2)
    wavelet = generate_ricker_wavelet(freq=25, dt=0.002, length=0.1)

    seismic_cube = np.zeros_like(reflection_coeff)
    for i in range(size[0]):
        for j in range(size[1]):
            seismic_cube[i, j, :] = convolve(reflection_coeff[i, j, :], wavelet, mode="same")

    seismic_norm = 2 * (seismic_cube - seismic_cube.min()) / (seismic_cube.max() - seismic_cube.min()) - 1

    return porosity, permeability, perm_norm, seismic_norm

# [3] 기초 통계 계산 함수
def basic_stats(data):
    return {
        "Min": np.min(data), "Max": np.max(data), "Mean": np.mean(data),
        "Std": np.std(data), "Median": np.median(data),
        "Q05": np.quantile(data, 0.05), "Q95": np.quantile(data, 0.95),
    }

# [4] 강화학습 의사결정 다목적 최적화 환경 클래스
class CCS_Decision_Maker_Env(gym.Env):
    def __init__(self, means, stds, weight_econ=0.5):
        super().__init__()
        self.means, self.stds = means, stds
        self.w_econ = weight_econ
        self.w_safe = 1.0 - weight_econ
        self.P_limit = 45.0

        # Action: 주입 조절율 (0.5 ~ 1.5)
        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(1,), dtype=np.float32)
        # Observation: [현재 시간, 평균 압력, 탄소 가격, 리스크 마진]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_year = 0
        return self._get_obs(), {}

    def _get_obs(self):
        m, s = self.means[self.current_year], self.stds[self.current_year]
        p_975 = m + 1.96 * s
        margin = self.P_limit - p_975
        price = 50.0 * (1.05 ** self.current_year) # 연 5% 탄소세 상승 모델
        return np.array([self.current_year/30.0, m/50.0, price/150.0, margin/10.0], dtype=np.float32)

    def step(self, action):
        m_base, s_base = self.means[self.current_year], self.stds[self.current_year]
        p_next_975 = (m_base + 1.96 * s_base) * action[0]

        price = 50.0 * (1.05 ** self.current_year)
        reward_econ = (price * (action[0] - 0.5)) / 100.0 

        reward_safe = 0.0
        if p_next_975 > self.P_limit:
            reward_safe = ((p_next_975 - self.P_limit) ** 2) * 20.0

        reward = (self.w_econ * reward_econ) - (self.w_safe * reward_safe)

        self.current_year += 1
        done = self.current_year >= 30

        if p_next_975 > 50.0:
            reward -= 100.0
            done = True

        return self._get_obs(), float(reward), done, False, {}
