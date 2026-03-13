## :link: :1st_place_medal: ASK2026 : 딥러닝을 활용한 CO2 지중 저장소 안정성 평가

주제 : 딥러닝을 활용한 CO2 지중 저장소 안정성 평가

* **저자 : 정지성, 김민지, 김주연, 오준석, 김경민, 김영균**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Reinforcement Learning](https://img.shields.io/badge/RL-PPO-blueviolet.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[cite_start]본 프로젝트는 탄소중립 실현을 위한 **CO2 지중 저장 기술(CCS)**의 안정성을 평가하고 최적의 주입 시나리오를 도출하기 위해 **물리 정보 신경망(PINNs)**, **불확실성 정량화(PCE)**, 그리고 **강화학습(PPO)**을 통합한 차세대 의사결정 시스템을 제안합니다[cite: 271, 284].

---

## 📝 01. 요약 (Abstract)
> [cite_start]글로벌 탄소 규제 강화에 따라 CCS 기술의 중요성이 대두되고 있으나, 기존 수치 해석 모델은 연산 부하가 크고 물리적 비일관성을 내포하는 한계가 있었습니다[cite: 269, 270]. [cite_start]본 연구는 물리 법칙을 내재화한 PINNs를 기반으로 연산 효율을 확보하고, 지질학적 불확실성을 확률적으로 정량화하여 임계 압력 내에서 경제적 수익을 극대화하는 동적 주입 제어 모델을 구축하였습니다[cite: 271, 285].


---

## 📌 02. 합성 데이터셋 생성 및 전처리 (Synthetic Dataset)
[cite_start]현장의 데이터 부족 한계를 극복하기 위해 3D 지진파 데이터와 1D 시추 로그를 결합한 고정밀 합성 데이터셋을 생성하였습니다[cite: 286].

* [cite_start]**공간 연속성 확보:** 가우시안 랜덤 필드(GRF) 기법을 활용하여 지층 특성을 재현하였습니다[cite: 288].
* [cite_start]**현장성 반영:** 노르웨이 **Sleipner 프로젝트**의 실제 저류층 데이터를 기반으로 보수적인 물성 범위를 설계하였습니다[cite: 290].
* [cite_start]**전처리:** 지진파 데이터는 선형 정규화([-1, 1]), 시추 로그는 Z-score 정규화를 적용하여 학습 안정성을 높였습니다[cite: 291].

---

## 📌 03. 물리 정보 신경망 설계 (PINNs Design)
[cite_start]저류층의 물리적 지배 방정식을 손실 함수에 포함하여 데이터가 제한된 환경에서도 타당한 예측이 가능하도록 설계하였습니다[cite: 292, 293].

* [cite_start]**통합 손실 함수 ($Loss_{total}$):** 데이터 정합성($Loss_{data}$), 질량 보존 및 Darcy 법칙 준수($Loss_{physics}$), 그리고 지반 안전 패널티($Loss_{penalty}$)를 결합하였습니다[cite: 294, 296, 297].
* [cite_start]**신경망 구조:** 5층 규모의 MLP와 물리 방정식의 미분 계산을 보장하는 Hyperbolic Tangent 활성화 함수를 채택하였습니다[cite: 299].

---

## 📌 04. 불확실성 정량화 설계 (Uncertainty Quantification)
[cite_start]지질학적 변수의 불확실성을 확률적으로 평가하기 위해 **다항식 카오스 전개(PCE)** 기반 대리 모델을 구축하였습니다[cite: 300, 301].

* [cite_start]**효율적 추론:** 복잡한 시뮬레이션 대신 직교 다항식의 선형 결합을 통해 짧은 시간 내 신뢰도 높은 예측을 수행합니다[cite: 301, 304].
* **안정성 기준:** 덮개암 파손 및 유도 지진 방지를 위해 임계 압력을 **45MPa**로 설정하여 위험도를 진단하였습니다[cite: 305, 306].

---

## 📌 05. 동적 주입 최적화 설계 (PPO Optimization)
지질학적 변동성과 시장 환경을 동시에 고려하여 최적의 주입량을 결정하는 강화학습 모델을 구축하였습니다[cite: 307, 308].

* [cite_start]**상태(State):** 저류층 평균 압력, 실시간 탄소 가격, 임계 압력 여유도 등을 인지합니다[cite: 309].
* [cite_start]**행동(Action):** 기준 유량 대비 조절 비율(0.5~1.5)을 통해 CO2 주입량을 직접 제어합니다[cite: 311].
* **보상(Reward):** 경제적 수익($R_{profit}$)을 추구하되, 임계 압력 초과 시 지수적으로 가중되는 패널티($R_{penalty}$)를 부여합니다[cite: 312, 313].

---

## 📝 06. 결론 (Conclusion)
본 모델은 기존 수치 해석의 시간적 제약을 해소하고 압도적인 예측 성능을 입증하였습니다[cite: 335, 340].

* [cite_start]**정확도 및 속도:** 미학습 데이터에서도 **$R^2$ 0.9950**의 높은 정확도와 **0.0015초**의 초고속 연산 속도를 달성하였습니다[cite: 332, 334, 339].
* [cite_start]**위험 저감:** 강화학습 제어를 통해 확률론적 위험 압력을 안전 범위 내로 정밀 제어하여 위험률을 **50% 이상 감소**시켰습니다[cite: 327, 338].

---

## 🔍 07. 향후 연구 (Future Work)
* **범용성 확보:** 특정 지역에 국한되지 않는 다양한 지질 조건에서의 범용 모델 개발[cite: 340].
* [cite_start]**실시간 모니터링:** 현장의 이상 징후를 즉각 탐지하고 제어하는 실시간 통합 의사결정 체계 고도화[cite: 340].

---

## 📄 08. 참고문헌 (References)
* [1] 이다영 외, "국내외 탄소 포집 및 저장 프로젝트 기술 동향," *한국에너지학회*, 2021.
* [2] 정은미, "탄소국경조정제도의 국내 산업계 파급효과," *전기저널*, 2021.
* [3] A. H. Emanuelsson, et al., "Deployment of CCS in the cement industry," *Int. J. Greenh. Gas Control*, 2025.
* [4] A. Zappone, et al., "Fault sealing and caprock integrity for CO2 storage," *Solid Earth*, 2021.
* [5] B. Saberali, et al., "Data-driven acceleration of reservoir simulations," *Geoenergy Sci. Eng.*, 2023.
* [6] 이상민, "물리정보 신경망과 응용 연구," *한국산학기술학회*, 2022.
* [7] M. GhojehBeyglou, "Geostatistical modeling of porosity," *J. Pet. Explor. Prod. Technol.*, 2021.
* [8] R. A. Chadwick, et al., "Monitoring of CO2 storage: the Sleipner field," *Geol. Soc. Lond. Spec. Publ.*, 2004.
* [9] M. Muskat, *The Flow of Homogeneous Fluids Through Porous Media*, McGraw-Hill, 1937.
* [10] M. Raissi, et al., "Physics-informed neural networks," *J. Comput. Phys.*, 2019.
* [11] N. Wiener, "The homogeneous chaos," *Am. J. Math.*, 1938.
* [12] R. G. Ghanem & P. D. Spanos, *Stochastic finite elements: A spectral approach*, Courier Corporation, 2003.

---

## 🧑‍💻 Citation

```bibtex
@article{jung2026deep,
  title={CO2 Underground Storage Stability Evaluation Using Deep Learning},
  author={Jisung Jung, Minji Kim, Juyeon Kim, JunSeok Oh, Gyeongmin Kim, Younggyun Kim},
  journal={Academic Research Project},
  year={2026}
}
