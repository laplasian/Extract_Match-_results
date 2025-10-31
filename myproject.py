import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from tqdm import tqdm
import os
import random
from __params__ import *

""" 
requirements:
    numpy>=1.26,<3.0
    scipy>=1.11,<2.0
    pandas>=2.2,<3.0
    matplotlib>=3.8,<4.0
    pymatgen>=2024.6.10,<2026.0
"""

# --- 1. ПАРАМЕТРЫ ВИРТУАЛЬНОГО ДИФРАКТОМЕТРА ---

# Длина волны, Cu K-alpha
WAVELENGTH_A1 = 1.54060  # K-alpha 1
WAVELENGTH_A2 = 1.54440  # K-alpha 2
INTENSITY_A1_FACTOR = 2.0 / 3.0
INTENSITY_A2_FACTOR = 1.0 / 3.0

# Диапазон 2-theta
TWO_THETA_MIN = 10.0
TWO_THETA_MAX = 90.0
STEP_SIZE = 0.01

# Коэффициент формы Шеррера (K)
K_SHERRER = 0.94

# Параметр шума
ADD_POISSON_NOISE = True


# --- 2. ГЛАВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ---

def generate_xrd(phases_info,
                 zero_shift=0.0,
                 inst_U=0.0004,
                 inst_V=-0.0002,
                 inst_W=0.0025,
                 inst_X=0.003,
                 inst_Y=0.0,
                 target_peak_counts=25000.0,
                 output_filename=None,
                 show_plot=False):
    """
    Генерирует дифрактограмму на основе списка фаз и их параметров.

    Args:
        phases_info (list): Список списков [...]
        zero_shift (float, optional): Глобальный сдвиг 2-Theta.

        ### <<< ИЗМЕНЕНИЕ: Новые аргументы
        inst_U, V, W, X, Y (float, optional): Параметры Кальоти (Caglioti)
                                              для инструментального уширения.
        target_peak_counts (float, optional): Целевая высота макс. пика
                                              (для контроля уровня шума).

        output_filename (str, optional): Путь для сохранения .xy файла.
        show_plot (bool, optional): Показать график.
    """

    calculator = XRDCalculator(wavelength=WAVELENGTH_A1)
    x_axis = np.arange(TWO_THETA_MIN, TWO_THETA_MAX, STEP_SIZE)
    y_total = np.zeros_like(x_axis)
    all_phase_profiles = []

    # --- Цикл по каждой фазе ---
    for phase_data in phases_info:
        cif_path, percentage, D_nm, strain_e, RIR = phase_data

        try:
            structure = Structure.from_file(cif_path)
        except Exception as e:
            print(f"Ошибка чтения CIF {cif_path}: {e}")
            continue

        pattern = calculator.get_pattern(structure, two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX))
        y_phase = np.zeros_like(x_axis)

        # --- Цикл по каждому пику (hkl) в фазе ---
        for two_theta_a1, intensity_total, hkl_indices in zip(pattern.x, pattern.y, pattern.hkls):

            # ... A. Расчет ФИЗИЧЕСКОГО уширения...
            theta_a1_rad = np.deg2rad(two_theta_a1 / 2.0)
            if D_nm is not None and D_nm > 0:
                D_A = D_nm * 10.0
                fwhm_L_phys_rad = (K_SHERRER * WAVELENGTH_A1) / (D_A * np.cos(theta_a1_rad))
            else:
                fwhm_L_phys_rad = 0.0
            if strain_e is not None and strain_e > 0:
                fwhm_G_phys_rad = 4 * strain_e * np.tan(theta_a1_rad)
            else:
                fwhm_G_phys_rad = 0.0
            fwhm_L_phys_deg = np.rad2deg(fwhm_L_phys_rad)
            fwhm_G_phys_deg = np.rad2deg(fwhm_G_phys_rad)

            # --- B. Расчет ИНСТРУМЕНТАЛЬНОГО уширения ---
            tan_theta = np.tan(theta_a1_rad)
            cos_theta = np.cos(theta_a1_rad)
            fwhm_G_inst_sq = inst_U * tan_theta ** 2 + inst_V * tan_theta + inst_W
            fwhm_G_inst_deg = np.sqrt(max(0, fwhm_G_inst_sq))
            fwhm_L_inst_deg = inst_X * tan_theta + inst_Y / cos_theta

            # ... C. Суммирование уширений  ...
            fwhm_G_total = np.sqrt(fwhm_G_phys_deg ** 2 + fwhm_G_inst_deg ** 2)
            fwhm_L_total = fwhm_L_phys_deg + fwhm_L_inst_deg

            # ... D. Генерация K-alpha1 / K-alpha2 ...
            sigma = fwhm_G_total / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            gamma = fwhm_L_total / 2.0
            if sigma < 1e-6 and gamma < 1e-6:
                continue

            sin_theta_a1 = np.sin(theta_a1_rad)
            sin_theta_a2 = (WAVELENGTH_A2 / WAVELENGTH_A1) * sin_theta_a1
            if sin_theta_a2 > 1.0: sin_theta_a2 = 1.0
            theta_a2_rad = np.arcsin(sin_theta_a2)
            two_theta_a2 = np.rad2deg(theta_a2_rad * 2.0)

            two_theta_a1_shifted = two_theta_a1 + zero_shift
            two_theta_a2_shifted = two_theta_a2 + zero_shift

            peak_profile_a1 = voigt_profile(x_axis - two_theta_a1_shifted, sigma, gamma)
            intensity_a1 = intensity_total * INTENSITY_A1_FACTOR
            y_phase += intensity_a1 * peak_profile_a1

            peak_profile_a2 = voigt_profile(x_axis - two_theta_a2_shifted, sigma, gamma)
            intensity_a2 = intensity_total * INTENSITY_A2_FACTOR
            y_phase += intensity_a2 * peak_profile_a2

        all_phase_profiles.append((y_phase, percentage, RIR))

    # --- 3. ФИНАЛИЗАЦИЯ (Смешивание, Масштабирование, Шум) ---
    for y_phase, percentage, RIR in all_phase_profiles:
        if RIR is None or RIR <= 0:
            RIR = 1.0
        y_total += y_phase * (percentage / 100.0) * RIR

    current_max_intensity = np.max(y_total)
    if current_max_intensity > 0:
        scale_factor = target_peak_counts / current_max_intensity
        y_total = y_total * scale_factor

    if ADD_POISSON_NOISE:
        y_total[y_total < 0] = 0
        y_total = np.random.poisson(y_total).astype(float)

    # --- 4. ВЫВОД ---
    if output_filename:
        dirname = os.path.dirname(output_filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savetxt(output_filename, np.column_stack((x_axis, y_total)),
                   fmt='%.4f', header="2Theta Intensity", comments='')

    if show_plot:
        plt.figure(figsize=(15, 6))
        plt.plot(x_axis, y_total, label="Синтезированная дифрактограмма")
        plt.xlabel("2-Theta [degrees]")
        plt.ylabel("Intensity [counts]")
        plt.title(f"Файл: {output_filename or 'Preview'}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    return x_axis, y_total

def generate_n_patterns(N_PATTERNS=1000, phases_info_base=[[None]], head=None):
    print(f"Запуск массовой генерации {N_PATTERNS} дифрактограмм...")

    OUTPUT_DIR = f"generated_patterns_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    head += "\nRate-% OKR-nm e\n"
    # Значения "по умолчанию" (модель Войта и другое). Около них варьируются значения
    try:
        for i in tqdm(range(N_PATTERNS), desc="Генерация..."):
            phases_info_base_ = phases_info_base.copy()
            current_params = {
                'inst_U': random.uniform(default_params['U'] * 0.8, default_params['U'] * 1.2),
                'inst_V': random.uniform(default_params['V'] * 0.8, default_params['V'] * 1.2),
                'inst_W': random.uniform(default_params['W'] * 0.8, default_params['W'] * 1.2),
                'inst_X': random.uniform(default_params['X'] * 0.8, default_params['X'] * 1.2),
                'inst_Y': default_params['Y'],  # Y почти всегда 0

                'zero_shift': random.uniform(-0.05, 0.05),  # Сдвиг от -0.05 до +0.05

                'target_peak_counts': random.uniform(1000, 4000)  # Разный уровень шума
            }

            for line in phases_info_base_:
                for j in range(1,4):
                    line[j] = random.uniform(line[j] * 0.8, line[j] * 1.2)


            output_file = os.path.join(OUTPUT_DIR, f"{i}.xy")

            # Вызываем функцию со всеми новыми параметрами
            generate_xrd(phases_info_base_,
                         zero_shift=current_params['zero_shift'],
                         inst_U=current_params['inst_U'],
                         inst_V=current_params['inst_V'],
                         inst_W=current_params['inst_W'],
                         inst_X=current_params['inst_X'],
                         inst_Y=current_params['inst_Y'],
                         target_peak_counts=current_params['target_peak_counts'],
                         output_filename=output_file,
                         show_plot=False
                         )
            for line in phases_info_base_:
                for k in range(1,4):
                    head = head + f"{line[k]} "
                head += "-- "
            head += "\n"


        print(f"\nГотово! {N_PATTERNS} файлов сохранено в папку '{OUTPUT_DIR}'.")

    except FileNotFoundError as e:
        print(f"\nОШИБКА: Не найден .cif файл: {e.filename}")
        print("Пожалуйста, убедитесь, что 'C.cif' и 'Co.cif' лежат рядом со скриптом.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

    with open("info.txt", "w") as f:
        f.write(head)