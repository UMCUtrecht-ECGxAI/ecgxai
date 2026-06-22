import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.collections import PatchCollection, LineCollection
import qrcode
from PIL import Image
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, Any
from matplotlib.lines import Line2D
import pandas as pd
from ecgxai.utils.collate import select_case_from_sample
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
from matplotlib.colors import ListedColormap

LAYOUT = {
    "3x4_1": [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11], [1]],
    "3x4": [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]],
    "6x2": [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
    "12x1": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
    "8x1": [[0], [1], [6], [7], [8], [9], [10], [11]],
    "SL": [[0]],
}

DEFAULT_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

DEFAULT_LABELS = {
    "": 0,
    "noise": 1,
    "p_wave_atrial_pacing": 2,
    "p_wave_atrial_rhythm": 3,
    "p_wave_no_av_conduction": 4,
    "p_wave_pac": 5,
    "p_wave_sinus_rhythm": 6,
    "qrs_atrial_fibrillation": 7,
    "qrs_atrial_flutter": 8,
    "qrs_atrial_pacing": 9,
    "qrs_atrial_rhythm": 10,
    "qrs_fusion": 11,
    "qrs_nodal_escape": 12,
    "qrs_pac": 13,
    "qrs_pvc": 14,
    "qrs_sinus_rhythm": 15,
    "qrs_svt": 16,
    "qrs_unknown": 17,
    "qrs_ventricular_escape": 18,
    "qrs_ventricular_pacing": 19,
    "qrs_vt": 20,
    "t_wave": 21,
}

DEFAULT_MAP_TO_ABBREV = {
    "": 0,
    "noise": "?",
    "p_wave_atrial_pacing": "P/",
    "p_wave_atrial_rhythm": "Pa",
    "p_wave_no_av_conduction": "P-",
    "p_wave_pac": "Pp",
    "p_wave_sinus_rhythm": "Ps",
    "qrs_atrial_fibrillation": "AF",
    "qrs_atrial_flutter": "AFl",
    "qrs_atrial_pacing": "A/",
    "qrs_atrial_rhythm": "AR",
    "qrs_fusion": "F",
    "qrs_nodal_escape": "J",
    "qrs_pac": "A",
    "qrs_pvc": "V",
    "qrs_sinus_rhythm": "N",
    "qrs_svt": "S",
    "qrs_unknown": "Q",
    "qrs_ventricular_escape": "V",
    "qrs_ventricular_pacing": "V/",
    "qrs_vt": "VT",
    "t_wave": None,
}


def generate_qr(data, qr_size=100):
    """Generate a QR code as a PIL image."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    # Resize QR code to desired size in pixels
    img = img.resize((qr_size, qr_size), Image.Resampling.LANCZOS)
    return img


def to12lead(waveform):
    out = np.zeros((12, waveform.shape[1]))
    out[0:2, :] = waveform[0:2, :]  # I and II
    out[2, :] = waveform[1, :] - waveform[0, :]  # III = II - I
    out[3, :] = -(waveform[0, :] + waveform[1, :]) / 2  # aVR = -(I + II)/2
    out[4, :] = waveform[0, :] - (waveform[1, :] / 2)  # aVL = I - II/2
    out[5, :] = waveform[1, :] - (waveform[0, :] / 2)  # aVF = II - I/2
    out[6:12, :] = waveform[2:8, :]  # V1 to V6
    return out


def plot_12lead_ecg(
    waveform,
    layoutid: str = "3x4_1",
    minor_grid: bool = True,
    qr_data: Optional[str] = None,
    dot: bool = False,
    overlays: Optional[np.ndarray] = None,
    grid: bool = True,
    info: bool = True,
    lead_name: bool = True,
    signal: bool = True,
    uV: bool = False,
    median_beat: bool = False,
    layout_color: Optional[str] = None,
    fs: Optional[int] = None
):
    """Plot the ecg signals inside the plotting area.
    waveform: numpy matrix with a 10 second ECG waveform in millivolts
              and shape n_channels x n_samples
    layout: ECG plotting layout, choices are 12x1, 6x2, 3x4 and 3x4_1
    minor_grid: add minor grid lines
    """
    factor = 1/1000 if uV else 1
    if isinstance(waveform, list):
        signals = []
        for i in waveform:
            if i.shape[0] == 8:
                signals.append(to12lead(i) * factor)
            elif i.shape[0] == 13:
                if sum(waveform[1], -1 * waveform[12]) == 0:
                    signals = waveform[:12] * factor
            else:
                signals.append(i * factor)
        samples = signals[0].shape[1]
        leads = signals.shape[0]
    else:
        if waveform.shape[0] == 8:
            signals = to12lead(waveform) * factor
        elif waveform.shape[0] == 13:
            waveform[1] = waveform[12] 
            signals = waveform[:12] * factor
        else:
            signals = waveform * factor
        samples = signals.shape[1]
        leads = signals.shape[0]
    if median_beat:
        layout = LAYOUT[layoutid]
        columns_med = len(layout[0])

        paper_w, paper_h = int(297 * samples * columns_med / 5000), int(250 * leads / 12)

        # Dimensions in mm of plot area
        width = int(250 * samples * columns_med / 5000)
        height = int(210 * leads / 12)
        margin_left = (paper_w - width) // 2
        margin_bottom = 10 * (leads / 12)
        left = margin_left / paper_w
        right = left + width / paper_w
        bottom = margin_bottom / paper_h
        top = bottom + height / paper_h

        sampling_frequency = int(samples / 10)
        if fs:
            sampling_frequency = fs
        duration = samples * columns_med / sampling_frequency
        mm_s = width / duration
        mm_mv = 10
    else:
        layout = LAYOUT[layoutid]
        rows = len(layout)

        paper_w, paper_h = int(280 * samples / 5000), int(250 * leads / 6)

        # Dimensions in mm of plot area
        width = int(250 * samples / 5000)
        height = int(210 * leads / 12)
        margin_left = (paper_w - width) // 2
        margin_bottom = 10 * (leads / 12)
        left = margin_left / paper_w
        right = left + width / paper_w
        bottom = margin_bottom / paper_h
        top = bottom + height / paper_h

        sampling_frequency = int(samples / 10)
        if fs:
            sampling_frequency = fs
        duration = samples / sampling_frequency
        mm_s = width / duration
        mm_mv = 10
    layout = LAYOUT[layoutid]
    rows = len(layout)
    # Init figure and axes
    fig = plt.figure(tight_layout=False)
    # Create a 4x4 grid
    if qr_data:
        gs = gridspec.GridSpec(5, 5, figure=fig)
        axes = fig.add_subplot(gs[1:, :])
    else:
        axes = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    axes.set_ylim([0, height])

    # We want to plot N points, where N=number of samples
    if median_beat:
        axes.set_xlim([0, (samples * columns_med)-1])
    else:
        axes.set_xlim([0, samples-1])
    channel_names = DEFAULT_LEAD_NAMES
    if signal:
        if isinstance(signals, list):
            color = ["black", "green", "blue"]
            for k in range(len(color)):
                for numrow, row in enumerate(layout):
                    columns = len(row)
                    row_height = height / rows

                    # Horizontal shift for lead labels and separators
                    h_delta = samples / columns

                    # Vertical shift of the origin
                    v_delta = round(
                        height * (1.0 - 1.0 / (rows * 2)) - numrow * (height / rows)
                    )

                    # Let's shift the origin on a multiple of 5 mm
                    v_delta = (v_delta + 2.5) - (v_delta + 2.5) % 5

                    # Lenght of a signal chunk
                    if median_beat:
                        chunk_size = int(samples)
                    else:
                        chunk_size = int(samples / len(row))
                    for numcol, signum in enumerate(row):
                        left = numcol * chunk_size
                        right = (1 + numcol) * chunk_size

                        # The signal chunk, vertical shifted and
                        # scaled by mm/mV factor
                        if k > 1:
                            sig = v_delta + mm_mv * abs(
                                signals[k - 2][signum][left:right]
                                - signals[k - 1][signum][left:right]
                            )
                            axes.plot(
                                list(range(left, right)),
                                sig,
                                clip_on=False,
                                linewidth=1,
                                color=color[k],
                                zorder=10,
                            )

                            meaning = channel_names[signum]

                            h = h_delta * numcol
                            v = v_delta + row_height / 2.6
                            plt.plot(
                                [h, h], [v - 3, v], lw=1, color=color[k], zorder=50
                            )

                            if lead_name:
                                axes.text(
                                    h + 40,
                                    v_delta + row_height / 3,
                                    meaning,
                                    zorder=50,
                                    fontsize=20,
                                )
                        else:
                            sig = v_delta + mm_mv * signals[k][signum][left:right]
                            axes.plot(
                                list(range(left, right)),
                                sig,
                                clip_on=False,
                                linewidth=1,
                                color=color[k],
                                zorder=10,
                            )

                            meaning = channel_names[signum]

                            h = h_delta * numcol
                            v = v_delta + row_height / 2.6
                            plt.plot(
                                [h, h], [v - 3, v], lw=1, color=color[k], zorder=50
                            )

                            axes.text(
                                h + 40,
                                v_delta + row_height / 3,
                                meaning,
                                zorder=50,
                                fontsize=10,
                            )
        else:
            mask_2d = None
            if layout_color is not None:
                mask_2d = np.asarray(layout_color)
                if mask_2d.ndim != 2:
                    raise ValueError("layout_color must be None or a 2D mask array.")
                if mask_2d.shape[0] != signals.shape[0]:
                    raise ValueError(
                        f"Mask first dim must match #channels. Got {mask_2d.shape[0]} vs {signals.shape[0]}."
                    )
                if mask_2d.shape[1] != signals.shape[1]:
                    raise ValueError(
                        f"Mask second dim must match #samples. Got {mask_2d.shape[1]} vs {signals.shape[1]}."
                    )

            for numrow, row in enumerate(layout):
                columns = len(row)
                row_height = height / rows

                h_delta = (samples / 1) if median_beat else (samples / columns)

                v_delta = round(
                    height * (1.0 - 1.0 / (rows * 2)) - numrow * (height / rows)
                )
                v_delta = (v_delta + 2.5) - (v_delta + 2.5) % 5

                chunk_size = int(samples) if median_beat else int(samples / len(row))

                for numcol, signum in enumerate(row):
                    left = numcol * chunk_size
                    right = (1 + numcol) * chunk_size

                    if median_beat:
                        x = np.arange(left, right)
                        y = v_delta + mm_mv * signals[signum, :]
                        mask_1d = None if mask_2d is None else mask_2d[signum, :]
                    else:
                        x = np.arange(left, right)
                        y = v_delta + mm_mv * signals[signum][left:right]
                        mask_1d = (
                            None if mask_2d is None else mask_2d[signum, left:right]
                        )

                    if mask_1d is None:
                        axes.plot(
                            x,
                            y,
                            clip_on=False,
                            linewidth=1,
                            color="#001201",
                            zorder=10,
                        )
                    else:
                        plot_masked_signal(
                            axes,
                            x=x,
                            y=y,
                            mask=mask_1d,
                            color0="#008D13",
                            color1="#001201",
                            linewidth=1,
                            zorder=10,
                            clip_on=False,
                        )

                    meaning = channel_names[signum]
                    h = h_delta * numcol
                    v = v_delta + row_height / 2.6
                    plt.plot([h, h], [v - 3, v], lw=1, color="#001201", zorder=50)

                    if lead_name:
                        axes.text(
                            h + 25,
                            v_delta + row_height / 3 - 5,
                            meaning,
                            zorder=50,
                            fontsize=22,
                        )

    if minor_grid:
        axes.xaxis.set_minor_locator(plt.LinearLocator(width + 1))
        axes.yaxis.set_minor_locator(plt.LinearLocator(height + 1))

    axes.xaxis.set_major_locator(plt.LinearLocator(width // 5 + 1))
    axes.yaxis.set_major_locator(plt.LinearLocator(height // 5 + 1))

    color = {"minor": "#ff5333", "major": "#FC6039"}
    linewidth = {"minor": 0.1, "major": 0.3}

    for axe in "x", "y":
        for which in "major", "minor":
            if grid:
                axes.grid(
                    which=which,
                    axis=axe,
                    linestyle="-",
                    linewidth=linewidth[which],
                    color=color[which],
                )

            axes.tick_params(
                which=which,
                axis=axe,
                color=color[which],
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

        axes.set_xticklabels([])
        axes.set_yticklabels([])

    if dot:
        xmaj = axes.get_xticks(minor=False)
        ymaj = axes.get_yticks(minor=False)

        plus_size = 0.1  # half-length of + lines

        for x in xmaj:
            for y in ymaj:
                # horizontal line
                axes.plot(
                    [x - plus_size - 1, x + plus_size + 1],
                    [y, y],
                    color="blue",
                    linewidth=1,
                )
                # vertical line
                axes.plot(
                    [x, x],
                    [y - plus_size, y + plus_size],
                    color="blue",
                    linewidth=1,
                )
    if info:
        info_left = (
            f"Total time: {duration}s / Sampling frequency: {sampling_frequency} Hz"
        )
        plt.figtext(
            0.02, 0.02,
            info_left,
            fontsize=8,
            ha="left",
        )

        info_right = (
            f"Speed: {mm_s:.1f} mm/s | Gain: {mm_mv:.1f} mm/mV"
        )
        plt.figtext(
            0.98, 0.02,
            info_right,
            fontsize=8,
            ha="right",
        )

        # A4 size in inches
    if median_beat:
        fig.set_size_inches(11.69 * (samples * columns_med / 5000), 9.27 * (leads / 12))
    else:
        fig.set_size_inches(11.69 * (samples / 5000), 9.27 * (leads / 12))

    if qr_data:
        qr_img = generate_qr(qr_data, qr_size=200)  # adjust qr_size in pixels
        qr_arr = np.array(qr_img)
        axes = fig.add_subplot(gs[0, 4])
        axes.imshow(qr_arr)

    return fig, axes


def plot_masked_signal(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    color0: str = "#008D13",  # mask == 0
    color1: str = "#001201",  # mask == 1
    linewidth: float = 1.0,
    zorder: int = 10,
    clip_on: bool = False,
) -> None:
    """
    Plot a 1D signal where each segment is colored by a binary mask.
    mask length must equal x/y length; colors apply per segment using mask[:-1].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.asarray(mask).astype(int)

    if x.ndim != 1 or y.ndim != 1 or mask.ndim != 1:
        raise ValueError("x, y, mask must be 1D arrays.")
    if len(x) != len(y) or len(mask) != len(x):
        raise ValueError("x, y, mask must have the same length.")

    if len(x) < 2:
        ax.plot(
            x,
            y,
            linewidth=linewidth,
            color=(color1 if (mask[0] == 1) else color0),
            zorder=zorder,
            clip_on=clip_on,
        )
        return

    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    seg_mask = mask[:-1]
    seg_colors = np.where(seg_mask == 1, color1, color0)

    lc = LineCollection(
        segments,
        colors=seg_colors,
        linewidths=linewidth,
        zorder=zorder,
        clip_on=clip_on,
    )
    ax.add_collection(lc)


def plot_original_segmentation(
    ecg, segmentation, lead_names=DEFAULT_LEAD_NAMES, labels=DEFAULT_LABELS
):
    cmap = plt.cm.get_cmap("nipy_spectral", 21)
    cmap.set_under("white")
    num_channels = ecg.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(20, 1.5 * num_channels))

    for lead in range(num_channels):
        ax.plot(ecg[lead, :] - 4 * lead, color="black")
        ax.text(0.1, 1.25 - 4 * lead, lead_names[lead], zorder=50, fontsize=14)

        img = plt.imshow(
            np.argmax(segmentation[:, lead, :], axis=0)[np.newaxis, :],
            extent=[0, 5000, (lead * -4) - 2, (lead * -4) + 2],
            aspect="auto",
            alpha=0.6,
            cmap=cmap,
            vmin=1,
            vmax=22,
        )

    height = num_channels * 4
    ax.set_ylim(-(height - 2), 2)

    cbar = plt.colorbar(img, ticks=np.arange(1.5, 22.5, 1))
    cbar.ax.set_yticklabels(list(labels.keys())[1:])

    plt.show()


def plot_segmentation(
    ecg: np.ndarray,
    median_beat: np.ndarray,
    complex_df,
    noise_df,
    corrected_waveform: np.ndarray,
    fiducials: Optional[np.ndarray],
    time: Optional[list] = None,
    lead_names=DEFAULT_LEAD_NAMES,
    labels: Dict[str, int] = DEFAULT_LABELS,
    map_to_abbrev: Dict[str, Optional[str]] = DEFAULT_MAP_TO_ABBREV,
    fs: int = 500,
    paper_speed: float = 25.0,
    gain_mm_mv: float = 10.0,
    lead_spacing_mm: float = 20.0,
    ax: Optional[plt.Axes] = None,
    info: bool = True,
):
    num_channels, n_samples = ecg.shape
    med_len = median_beat.shape[1]

    total_samples = n_samples + med_len
    total_duration_s = n_samples / fs

    mm_per_sample = paper_speed / fs
    x_mm = np.arange(n_samples) * mm_per_sample

    total_height = num_channels * lead_spacing_mm
    half_height = lead_spacing_mm / 2
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    samples = n_samples

    paper_w, paper_h = int(297 * samples / 5000), int(250 *  num_channels / 12)

    width = int(250 * samples / 5000)
    height = int(210 *  num_channels / 12)
    margin_left = (paper_w - width) // 2
    margin_bottom = (paper_h - height) // 2
    left = margin_left / paper_w
    right = left + width / paper_w
    bottom = margin_bottom / paper_h
    top = bottom + height / paper_h

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for lead in range(num_channels):
        center = (num_channels - lead - 1) * lead_spacing_mm

        ax.plot(
            x_mm,
            center + gain_mm_mv * corrected_waveform[lead],
            color="0.7",
            lw=0.8,
            zorder=1,
        )

        ax.plot(x_mm, center + gain_mm_mv * ecg[lead], color="black", lw=1.0, zorder=2)

        med_x = x_mm[-1] + mm_per_sample * np.arange(med_len)
        ax.plot(
            med_x,
            center + gain_mm_mv * median_beat[lead],
            color="black",
            lw=1.0,
            zorder=2,
        )

        if fiducials is not None:
            idx = fiducials[~np.isnan(fiducials)].astype(int)
            idx = idx[(idx >= 0) & (idx < med_len)]
            ax.vlines(med_x[idx], center - 5, center + 5, color="0.5", lw=1)

        ax.text(
            x_mm[0] - 5, center, lead_names[lead], ha="right", va="center", fontsize=10
        )

        # ================== SEGMENTATION OVERLAYS ==================
        def generate_distinct_colors(n: int) -> ListedColormap:
            """
            Generate n visually distinct colors using HSV space.
            Ensures no repeated colors even if n > built-in cmap size.
            """
            hues = np.linspace(0, 1, n, endpoint=False)
            colors = plt.cm.hsv(hues)
            return ListedColormap(colors)


    cmap = generate_distinct_colors(len(labels))
    used_types = set()

    def draw_segment(onset, offset_samp, label):
        if np.isnan(onset) or np.isnan(offset_samp):
            return
        if label not in labels:
            return

        start = onset * mm_per_sample
        width = (offset_samp - onset) * mm_per_sample
        color = cmap(labels[label])

        for ld in range(num_channels):
            center = (num_channels - ld - 1) * lead_spacing_mm
            ax.add_patch(
                patches.Rectangle(
                    (start, center - half_height),
                    width,
                    lead_spacing_mm,
                    facecolor=color,
                    alpha=0.35,
                    lw=1,
                )
            )

    for _, wave in complex_df.iterrows():
        if isinstance(wave.get("p_type"), str) and "unspecified" not in wave["p_type"]:
            draw_segment(wave["p_onset"], wave["p_offset"], wave["p_type"])
            used_types.add(wave["p_type"])

        if (
            isinstance(wave.get("qrs_type"), str)
            and "unspecified" not in wave["qrs_type"]
        ):
            draw_segment(wave["qrs_onset"], wave["qrs_offset"], wave["qrs_type"])
            used_types.add(wave["qrs_type"])

        if "t_onset" in wave:
            draw_segment(wave["t_onset"], wave["t_offset"], "TTYPE_UNSPECIFIED")
            used_types.add("TTYPE_UNSPECIFIED")

        if "st_index" in wave:
            draw_segment(wave["st_index"], wave["st_index"] + 5, "ST_DEPRESSION")
            used_types.add("ST_DEPRESSION")

        if "base_index" in wave:
            draw_segment(wave["base_index"], wave["base_index"] + 5, "BASE_LINE")
            used_types.add("BASE_LINE")

    for _, noise in noise_df.iterrows():
        if isinstance(noise.get("lead"), str) and "unspecified" not in noise["lead"]:
            lead_idx = lead_names.index(noise["lead"])
            center = (num_channels - lead_idx - 1) * lead_spacing_mm
            start = noise["onset"] * mm_per_sample
            width = (noise["offset"] - noise["onset"]) * mm_per_sample

            ax.add_patch(
                patches.Rectangle(
                    (start, center - half_height),
                    width,
                    lead_spacing_mm,
                    facecolor="black",
                    alpha=0.35,
                    lw=1,
                )
            )

    # ================== CALIBRATION ==================
    calib_height_mm = gain_mm_mv
    calib_width_mm = 5.0
    calib_x0 = 0.0

    for lead in range(num_channels):
        center = (num_channels - lead - 1) * lead_spacing_mm
        ax.plot(
            [calib_x0, calib_x0, calib_x0 + calib_width_mm, calib_x0 + calib_width_mm],
            [center, center + calib_height_mm, center + calib_height_mm, center],
            color="#ff4c30",
            lw=2,
            zorder=5,
        )

    # ================== AXES / GRID ==================
    ax.set_xlim(0, total_samples * mm_per_sample)
    ax.set_ylim(-lead_spacing_mm, total_height)

    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

    ax.grid(which="major", color="#ff5333", lw=0.3)
    ax.grid(which="minor", color="#ff5333", lw=0.1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which="both", length=0)

    # ================== LEGEND ==================
    handles = []
    for t in sorted(used_types):
        if t in labels:
            handles.append(Line2D([0], [0], color=cmap(labels[t]), lw=6, label=t))

    if handles:
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(handles), 8),
            frameon=False,
            fontsize=12,
        )

    # ================== INFO TEXT ==================

    if info:
        info = f"Total time: {total_duration_s}s / Sampling frequency: {fs} Hz"
        plt.figtext(0.08, 0.065, info, fontsize=8)

        info = f"{paper_speed} mm/s {gain_mm_mv} mm/mV"
        plt.figtext(0.81, 0.065, info, fontsize=8)

    if time is not None and len(time) > 0:
        time_str = start_end_time_label(time)
        plt.figtext(0.5, 0.065, time_str, fontsize=8)

    # ================== FINAL SIZE (OLD BEHAVIOR) ==================
    fig.set_size_inches(
        11.69 * (total_samples * mm_per_sample / (5000 * mm_per_sample)),
        8.27 * ((total_height + lead_spacing_mm) / ((12 * lead_spacing_mm) +lead_spacing_mm)),
        forward=True,
    )

    return fig, ax


def start_end_time_label(time_values):
    if not time_values:
        raise ValueError("Empty time list")

    t0 = str(time_values[0])
    t1 = str(time_values[-1])

    start_time = t0.split("T")[1].split(".")[0]
    end_time = t1.split("T")[1].split(".")[0]

    return f"Time range: {start_time} → {end_time}"


class ContinuousSegmentationViewer:
    def __init__(
        self,
        dataset,
        window_seconds: float = 10.0,
        preload_radius: int = 1,
    ):
        self.dataset = dataset
        self.radius = preload_radius

        first = select_case_from_sample(self.dataset.__getitem__(0))
        self.fs = first["samplebase"]
        self.seg_len = first["waveform"].shape[1]

        self.window_samples = int(window_seconds * self.fs)
        self.total_samples = self.dataset.__len__() * self.seg_len
        self.times = self.dataset.__gettime__(0)
        self.start_time = self.times["start_time"]
        self.end_time = self.times["end_time"]

        self.total_duration = int(
            (self.end_time - self.start_time) / np.timedelta64(1, "s")
        )

        self.window_seconds = int(self.window_samples / self.fs)
        self.max_time = self.total_duration - self.window_seconds

        self.current_sample = 0
        self.loaded_center = None
        self.buffer = None

        self.fig, self.ax = plt.subplots()

        # --- create slider ---
        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.max_time,
            step=1,
            continuous_update=True,
            description="Time",
            layout=widgets.Layout(width="95%"),
        )

        self.slider.observe(self._on_slider_widget, names="value")

        # --- create label ---
        self.time_label = widgets.Label()

        # --- container goes HERE ---
        container = widgets.VBox(
            [
                self.fig.canvas,
                self.time_label,
                self.slider,
            ]
        )
        display(container)
        plt.close(self.fig)

        # --- initial draw ---
        self.time_label.value = str(self.start_time)
        self._update(force=True)

    # ======================================================
    # Buffer assembly (3 contiguous segments)
    # ======================================================
    def _on_slider_widget(self, change):
        t_sec = int(change["new"])

        # absolute datetime
        current_time = self.start_time + np.timedelta64(t_sec, "s")

        # update label
        self.time_label.value = str(current_time)

        # convert to samples
        sample_idx = int(t_sec * self.fs)
        sample_idx = min(
            max(sample_idx, 0),
            self.total_samples - self.window_samples,
        )

        self.current_sample = sample_idx
        self._update()

    def _assemble_buffer(self, center_seg: int):
        segments = []

        for s in range(center_seg - self.radius, center_seg + self.radius + 1):
            if 0 <= s < len(self.dataset):
                segments.append(select_case_from_sample(self.dataset.__getitem__[s]))

        wf, corr, times = [], [], []
        complex_dfs, noise_dfs = [], []

        SHIFT_COLS = {
            "onset",
            "offset",
            "p_onset",
            "p_offset",
            "qrs_onset",
            "qrs_offset",
            "t_onset",
            "t_offset",
        }

        for i, seg in enumerate(segments):
            wf.append(seg["waveform"])
            corr.append(seg["waveform"])
            times.extend(seg["time"])

            shift = i * self.seg_len

            # ================= complex_df =================
            cdict = seg.get("complex_df", {})
            if isinstance(cdict, dict) and cdict:
                dfc = pd.DataFrame(cdict)

                for col in SHIFT_COLS:
                    if col in dfc.columns:
                        # works for numpy arrays with NaNs
                        dfc[col] = dfc[col] + shift

                complex_dfs.append(dfc)

            # ================= noise_df =================
            ndict = seg.get("noise_df", {})
            if isinstance(ndict, dict) and ndict:
                dfn = pd.DataFrame(ndict)
                dfn["onset"] = dfn["onset"] + shift
                dfn["offset"] = dfn["offset"] + shift
                noise_dfs.append(dfn)

        return {
            "waveform": np.concatenate(wf, axis=1),
            "corrected_waveform": np.concatenate(corr, axis=1),
            "time": times,
            "complex_df": (
                pd.concat(complex_dfs, ignore_index=True)
                if complex_dfs
                else pd.DataFrame()
            ),
            "noise_df": (
                pd.concat(noise_dfs, ignore_index=True) if noise_dfs else pd.DataFrame()
            ),
            "median_beat": segments[0]["median_beat"],
            "fiducials": segments[0].get("fiducials"),
            "samplebase": segments[0]["samplebase"],
            "center_seg": center_seg,
        }

    # ======================================================
    # Window extraction (global → local mapping)
    # ======================================================

    def _extract_window(self, global_sample: int):
        center_seg = global_sample // self.seg_len

        if center_seg != self.loaded_center:
            self.buffer = self._assemble_buffer(center_seg)
            self.loaded_center = center_seg

        buffer_start = (center_seg - self.radius) * self.seg_len
        local_start = global_sample - buffer_start
        local_end = local_start + self.window_samples

        wf = self.buffer["waveform"][:, local_start:local_end]
        corr = self.buffer["corrected_waveform"][:, local_start:local_end]
        time_window = self.buffer["time"][local_start:local_end]

        complex_df = self.buffer["complex_df"]
        noise_df = self.buffer["noise_df"]

        # ---------- filter rows ----------
        complex_df = complex_df[
            (complex_df["offset"] > local_start) & (complex_df["onset"] < local_end)
        ].copy()

        noise_df = noise_df[
            (noise_df["offset"] > local_start) & (noise_df["onset"] < local_end)
        ].copy()

        # ---------- shift ALL timing columns ----------
        for col in (
            "onset",
            "offset",
            "p_onset",
            "p_offset",
            "qrs_onset",
            "qrs_offset",
            "t_onset",
            "t_offset",
        ):
            if col in complex_df.columns:
                complex_df[col] = complex_df[col] - local_start

        if not noise_df.empty:
            noise_df["onset"] = noise_df["onset"] - local_start
            noise_df["offset"] = noise_df["offset"] - local_start

        return wf, corr, time_window, complex_df, noise_df

    # ======================================================
    # Plot update
    # ======================================================

    def _update(self, force=False):
        self.ax.clear()

        wf, corr, time_window, complex_df, noise_df = self._extract_window(
            self.current_sample
        )

        plot_segmentation(
            ecg=wf,
            corrected_waveform=corr,
            median_beat=self.buffer["median_beat"],
            fiducials=self.buffer["fiducials"],
            complex_df=complex_df,
            noise_df=noise_df,
            time=time_window,
            fs=self.buffer["samplebase"],
            ax=self.ax,
        )

        self.fig.canvas.draw_idle()


class FastContinuousSegmentationViewer:
    """
    High-performance ECG segmentation viewer with:
      - time-based slider
      - play / pause
      - segmentation toggle
      - moving datetime label

    Grid is STATIC.
    ECG lines, median, segments, noise are ANIMATED.
    """
    def __init__(
        self,
        dataset,
        labels=DEFAULT_LABELS,
        lead_names=DEFAULT_LEAD_NAMES,
        window_seconds=10.0,
        preload_radius=4,
        paper_speed=25.0,
        gain_mm_mv=10.0,
        lead_spacing_mm=20.0,
        enable_widgets=True,
    ):
        self.dataset = dataset
        self.labels = labels
        self.lead_names = lead_names
        self.radius = preload_radius

        first = select_case_from_sample(dataset.__getitem__(0))
        self.fs = first["samplebase"]
        self.seg_len = first["waveform"].shape[1]
        self.num_channels = first["waveform"].shape[0]

        self.window_samples = int(window_seconds * self.fs)
        self.total_samples = len(dataset) * self.seg_len

        self.paper_speed = paper_speed
        self.mm_per_sample = paper_speed / self.fs
        self.gain = gain_mm_mv
        self.lead_spacing = lead_spacing_mm
        self.half_height = lead_spacing_mm / 2

        self.lead_to_idx = {n: i for i, n in enumerate(self.lead_names)}
        self.cmap = plt.cm.get_cmap("tab20", max(labels.values()) + 1)

        self.current_sample = 0
        self.loaded_center = None
        self.buffer = None
        self.legend = None

        # --- playback ---
        self.show_segments = True
        self._playing = False
        self._play_thread = None
        self.play_interval = 0.05

        # --- time metadata ---
        times = self.dataset.__gettime__(0)
        self.start_time = times["start_time"]
        self.end_time = times["end_time"]

        self.total_duration = int(
            (self.end_time - self.start_time) / np.timedelta64(1, "s")
        )
        self.window_seconds = int(self.window_samples / self.fs)
        self.max_time = max(0, self.total_duration - self.window_seconds)

        # ==================================================
        # Figure
        # ==================================================
        self.fig, self.ax = plt.subplots()
        self._init_axes()
        self._init_lines()
        self._init_lead_names()
        self._init_patches()
        self._init_info_text()

        self.fig.set_size_inches(
            11.69 * (self.window_samples / 5000),
            8.27,
            forward=True,
        )

        # ==================================================
        # Widgets
        # ==================================================
        if enable_widgets:
            self._init_widgets()
            self._on_slider_widget({"new": 0})

    # ======================================================
    # Widgets
    # ======================================================
    def _init_widgets(self):
        self.start_label = widgets.Label(self._fmt_dt(self.start_time))
        self.end_label = widgets.Label(self._fmt_dt(self.end_time))

        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.max_time,
            step=1,
            continuous_update=True,
            readout=False,
            layout=widgets.Layout(flex="1 1 auto"),
        )
        self.slider.observe(self._on_slider_widget, names="value")

        self.time_label = widgets.HTML()
        self.time_label_container = widgets.Box(
            [self.time_label],
            layout=widgets.Layout(width="100%", position="relative", height="20px"),
        )

        slider_row = widgets.HBox(
            [self.start_label, self.slider, self.end_label],
            layout=widgets.Layout(align_items="center"),
        )

        self.play_button = widgets.ToggleButton(
            value=False, description="▶ Play", icon="play"
        )
        self.segment_button = widgets.ToggleButton(
            value=True, description="Segments ON", icon="eye"
        )

        self.play_button.observe(self._on_play_toggle, names="value")
        self.segment_button.observe(self._on_segment_toggle, names="value")

        controls = widgets.HBox(
            [self.play_button, self.segment_button],
            layout=widgets.Layout(justify_content="center"),
        )

        display(
            widgets.VBox(
                [
                    self.fig.canvas,
                    self.time_label_container,
                    slider_row,
                    controls,
                ]
            )
        )

    # ======================================================
    # Formatting
    # ======================================================
    def _fmt_dt(self, t):
        return np.datetime_as_string(t, unit="s").replace("T", " ")

    # ======================================================
    # Slider
    # ======================================================
    def _on_slider_widget(self, change):
        t_sec = int(change["new"])
        current_time = self.start_time + np.timedelta64(t_sec, "s")

        frac = t_sec / self.max_time if self.max_time else 0.0
        left = min(max(int(frac * 100), 2), 98)

        self.time_label.value = (
            f"<div style='position:absolute; left:{left}%; "
            f"transform:translateX(-50%); font-size:12px;'>"
            f"{self._fmt_dt(current_time)}</div>"
        )

        self.current_sample = min(
            int(t_sec * self.fs),
            self.total_samples - self.window_samples,
        )
        self._update()

    # ======================================================
    # Play / Pause
    # ======================================================
    def _on_play_toggle(self, change):
        self._playing = change["new"]

        if self._playing:
            self.play_button.description = "⏸ Pause"
            self.play_button.icon = "pause"
            self._start_playback()
        else:
            self.play_button.description = "▶ Play"
            self.play_button.icon = "play"

    def _start_playback(self):
        if self._play_thread and self._play_thread.is_alive():
            return

        def run():
            while self._playing:
                if self.slider.value >= self.slider.max:
                    self.play_button.value = False
                    break
                self.slider.value += 1
                time.sleep(self.play_interval)

        self._play_thread = threading.Thread(target=run, daemon=True)
        self._play_thread.start()

    # ======================================================
    # Segment toggle
    # ======================================================
    def _on_segment_toggle(self, change):
        self.show_segments = change["new"]
        self.segment_button.description = (
            "Segments ON" if self.show_segments else "Segments OFF"
        )
        self.segment_button.icon = "eye" if self.show_segments else "eye-slash"

        # legend needs full redraw
        self.legend = None
        self._init_static_view()
        self._update()

    # ======================================================
    # Axes / lines
    # ======================================================
    def _init_axes(self):
        ax = self.ax
        ax.set_ylim(-10, self.num_channels * self.lead_spacing)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(which="major", color="#ff5333", lw=0.3)
        ax.grid(which="minor", color="#ff5333", lw=0.1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(which="both", length=0)

    def _init_lines(self):
        self.ecg_lines = []
        self.corrected_lines = []
        self.median_lines = []

        for _ in range(self.num_channels):
            self.corrected_lines.append(
                self.ax.plot([], [], color="0.7", lw=0.8, animated=True)[0]
            )
            self.ecg_lines.append(
                self.ax.plot([], [], color="black", lw=1.0, animated=True)[0]
            )
            self.median_lines.append(
                self.ax.plot([], [], color="black", lw=1.0, animated=True)[0]
            )

    def _init_lead_names(self):
        self.lead_texts = []
        for ld in range(self.num_channels):
            y = (self.num_channels - ld - 1) * self.lead_spacing
            self.lead_texts.append(
                self.ax.text(-12, y, self.lead_names[ld], ha="right", va="center")
            )

    def _init_patches(self):
        self.seg_patches = []
        self.noise_patches = []

    def _init_info_text(self):
        self.info_left = self.fig.text(0.08, 0.025, "", fontsize=8)
        self.info_center = self.fig.text(0.5, 0.025, "", fontsize=8, ha="center")
        self.info_right = self.fig.text(0.81, 0.025, "", fontsize=8)
    # ======================================================
    # Helpers
    # ======================================================
    def _ensure_patches(self, store, n):
        while len(store) < n:
            r = patches.Rectangle((0, 0), 0, 0, alpha=0.35, animated=False)
            self.ax.add_patch(r)
            store.append(r)


    # ======================================================
    # Segments
    # ======================================================
    def _update_segments(self, df):
        for r in self.seg_patches:
            r.set_visible(False)

        if not self.show_segments or df.empty:
            return set(), []

        segs = []

        for _, row in df.iterrows(): 
            if isinstance(row.get("p_type"), str) and "unspecified" not in row["p_type"]: 
                segs.append((row["p_onset"], row["p_offset"], row["p_type"])) 
            if isinstance(row.get("qrs_type"), str) and "unspecified" not in row["qrs_type"]: 
                segs.append((row["qrs_onset"], row["qrs_offset"], row["qrs_type"])) 
            if "t_onset" in row and not pd.isna(row["t_onset"]): 
                segs.append((row["t_onset"], row["t_offset"], "t_wave"))

        self._ensure_patches(self.seg_patches, len(segs))

        used = set()
        visible = []

        for r, (onset, offset, label) in zip(self.seg_patches, segs):
            if label not in self.labels:
                continue

            r.set_xy((onset * self.mm_per_sample, -self.half_height))
            r.set_width((offset - onset) * self.mm_per_sample)
            r.set_height(self.num_channels * self.lead_spacing - 1)
            r.set_facecolor(self.cmap(self.labels[label]))
            r.set_visible(True)

            used.add(label)
            visible.append(r)

        return used, visible


    # ======================================================
    # Noise
    # ======================================================
    def _update_noise(self, df):
        for r in self.noise_patches:
            r.set_visible(False)

        if not self.show_segments or df.empty:
            return []

        self._ensure_patches(self.noise_patches, len(df))

        visible = []

        for r, (_, row) in zip(self.noise_patches, df.iterrows()):
            ld = self.lead_to_idx.get(row.get("lead"))
            if ld is None:
                continue

            y = (self.num_channels - ld - 1) * self.lead_spacing
            r.set_xy((row["onset"] * self.mm_per_sample, y - self.half_height))
            r.set_width((row["offset"] - row["onset"]) * self.mm_per_sample)
            r.set_height(self.lead_spacing)
            r.set_facecolor(self.cmap(self.labels["noise"]))
            r.set_visible(True)

            visible.append(r)

        return visible


    # ======================================================
    # Main update (blitted)
    # ======================================================
    def _update(self):
        wf, corr, median, time, cdf, ndf = self._extract_window()

        self.fig.canvas.restore_region(self._background)

        n = wf.shape[1]
        x = np.arange(n) * self.mm_per_sample
        mx = x[-1] + self.mm_per_sample * np.arange(median.shape[1])

        for ld in range(self.num_channels):
            y = (self.num_channels - ld - 1) * self.lead_spacing

            self.corrected_lines[ld].set_data(x, y + self.gain * corr[ld])
            self.ecg_lines[ld].set_data(x, y + self.gain * wf[ld])
            self.median_lines[ld].set_data(mx, y + self.gain * median[ld])

            self.ax.draw_artist(self.corrected_lines[ld])
            self.ax.draw_artist(self.ecg_lines[ld])
            self.ax.draw_artist(self.median_lines[ld])

        used, seg_patches = self._update_segments(cdf)
        noise_patches = self._update_noise(ndf)

        for r in seg_patches:
            self.ax.draw_artist(r)
        for r in noise_patches:
            self.ax.draw_artist(r)

        self._update_legend(used, bool(len(noise_patches)))
        self._update_info_text(time, n)

        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    # ======================================================
    # Buffer
    # ======================================================
    def _assemble_buffer(self, center_seg):
        segments = []
        for s in range(center_seg - self.radius, center_seg + self.radius + 1):
            if 0 <= s < len(self.dataset):
                segments.append(select_case_from_sample(self.dataset.__getitem__(s)))

        wf, corr, times = [], [], []
        complex_dfs, noise_dfs = [], []

        SHIFT_COLS = {
            "onset",
            "offset",
            "p_onset",
            "p_offset",
            "qrs_onset",
            "qrs_offset",
            "t_onset",
            "t_offset",
        }

        for i, seg in enumerate(segments):
            wf.append(seg["waveform"])
            corr.append(seg["waveform"])
            times.extend(seg["time"])

            shift = i * self.seg_len

            # ==================================================
            # complex_df — SAME SEMANTICS AS BEFORE
            # ==================================================
            cdf = seg.get("complex_df")

            dfc = None
            if isinstance(cdf, pd.DataFrame):
                if not cdf.empty:
                    dfc = cdf.copy()

            elif isinstance(cdf, dict) and cdf:
                # dict-of-dicts → rows
                first_val = next(iter(cdf.values()))
                if isinstance(first_val, dict):
                    dfc = pd.DataFrame.from_records(list(cdf.values()))
                else:
                    # dict-of-lists → columns
                    dfc = pd.DataFrame(cdf)

            if dfc is not None and not dfc.empty:
                for c in SHIFT_COLS & set(dfc.columns):
                    dfc[c] = dfc[c] + shift
                complex_dfs.append(dfc)

            # ==================================================
            # noise_df — SAME AS ORIGINAL
            # ==================================================
            ndf = seg.get("noise_df")
            dfn = None

            if isinstance(ndf, pd.DataFrame):
                if not ndf.empty:
                    dfn = ndf.copy()

            elif isinstance(ndf, dict) and ndf:
                first_val = next(iter(ndf.values()))
                if isinstance(first_val, dict):
                    dfn = pd.DataFrame.from_records(list(ndf.values()))
                else:
                    dfn = pd.DataFrame(ndf)

            if dfn is not None and not dfn.empty:
                if "onset" in dfn:
                    dfn["onset"] += shift
                if "offset" in dfn:
                    dfn["offset"] += shift
                noise_dfs.append(dfn)

        return {
            "waveform": np.concatenate(wf, axis=1),
            "corrected": np.concatenate(corr, axis=1),
            "time": times,
            "complex_df": (
                pd.concat(complex_dfs, ignore_index=True)
                if complex_dfs
                else pd.DataFrame()
            ),
            "noise_df": (
                pd.concat(noise_dfs, ignore_index=True) if noise_dfs else pd.DataFrame()
            ),
            "median": segments[0]["median_beat"],
        }

    def _extract_window(self):
        center = self.current_sample // self.seg_len
        if center != self.loaded_center:
            self.buffer = self._assemble_buffer(center)
            self.loaded_center = center

        start = self.current_sample - (center - self.radius) * self.seg_len
        end = start + self.window_samples

        wf = self.buffer["waveform"][:, start:end]
        corr = self.buffer["corrected"][:, start:end]
        time = self.buffer["time"][start:end]
        median = self.buffer["median"]

        cdf = self.buffer["complex_df"]
        ndf = self.buffer["noise_df"]

        if not cdf.empty:
            # main interval overlap (unchanged)
            cdf = cdf[(cdf["offset"] > start) & (cdf["onset"] < end)].copy()

            # shift everything
            for c in cdf.columns:
                if c.endswith("onset") or c.endswith("offset"):
                    cdf[c] -= start

            # invalidate sub-intervals that do NOT overlap the window
            def _invalidate(df, on, off):
                if on in df and off in df:
                    mask = (df[off] > 0) & (df[on] < self.window_samples)
                    df.loc[~mask, [on, off]] = np.nan

            _invalidate(cdf, "p_onset", "p_offset")
            _invalidate(cdf, "qrs_onset", "qrs_offset")
            _invalidate(cdf, "t_onset", "t_offset")

        if not ndf.empty:
            ndf = ndf[(ndf["offset"] > start) & (ndf["onset"] < end)].copy()
            ndf["onset"] -= start
            ndf["offset"] -= start

        return wf, corr, median, time, cdf, ndf

    # ======================================================
    # Legend / Info
    # ======================================================
    def _update_legend(self, used, has_noise):
        if self.legend:
            self.legend.remove()
        if has_noise:
            used.add("noise")
        handles = [
            Line2D([0], [0], lw=6, color=self.cmap(self.labels[l]), label=l)
            for l in sorted(used, key=lambda x: self.labels[x])
        ]

        if handles:
            self.legend = self.ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=False,
                ncol=min(len(handles), 8),
            )

    def _update_info_text(self, time, n):
        self.info_left.set_text(f"{n / self.fs:.1f}s @ {self.fs} Hz")
        self.info_right.set_text(f"{self.paper_speed} mm/s  {self.gain} mm/mV")

        if time:
            t0, t1 = str(time[0]), str(time[-1])
            self.info_center.set_text(
                f"{t0.split('T')[1].split('.')[0]} → {t1.split('T')[1].split('.')[0]}"
            )

import time
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from plotly.colors import qualitative
from contextlib import nullcontext


class FastContinuousSegmentationViewerPlotly:
    """
    Plotly equivalent of FastContinuousSegmentationViewer.

    - Static grid
    - Animated ECG / median
    - Segment + noise overlays
    - Slider + play/pause
    """

    # ======================================================
    # INIT
    # ======================================================
    def __init__(
        self,
        dataset,
        labels = DEFAULT_LABELS,
        lead_names = DEFAULT_LEAD_NAMES,
        window_seconds=10.0,
        preload_radius=2,
        paper_speed=25.0,
        gain_mm_mv=10.0,
        lead_spacing_mm=20.0,
        play_interval=0.00001,
        fast_mode=True,
    ):
        self.dataset = dataset
        self.labels = labels
        self.lead_names = lead_names
        self.radius = preload_radius
        self.play_interval = play_interval

        first = dataset[0]
        self.fs = first["samplebase"]
        self.seg_len = first["waveform"].shape[1]
        self.med= first["median_beat"].shape[1]
        self.num_channels = first["waveform"].shape[0]

        self.window_samples = int(window_seconds * self.fs)
        self.figure_samples = self.window_samples + self.med
        self.total_samples = len(dataset) * self.seg_len

        self.paper_speed = paper_speed
        self.mm_per_sample = paper_speed / self.fs
        self.gain = gain_mm_mv
        self.lead_spacing = lead_spacing_mm
        self.half_height = lead_spacing_mm / 2

        self.current_sample = 0
        self.loaded_center = None
        self.buffer = None
        self.fast_mode = fast_mode

         # --- playback ---
        self.show_segments = True
        self._playing = False
        self._play_thread = None
        self.play_interval = 0.0001

        palette = qualitative.Dark24
        self.color_map = {
            k: palette[v % len(palette)] for k, v in labels.items()
        }
        self._legend_traces = {}
        self._update_fn = self._update_fast if self.fast_mode else self._update


        times = self.dataset.__gettime__(0)
        self.start_time = times["start_time"]
        self.end_time = times["end_time"]

        self.total_duration = int(
            (self.end_time - self.start_time) / np.timedelta64(1, "s")
        )
        self.window_seconds = int(self.window_samples / self.fs)
        self.max_time = max(0, self.total_duration - self.window_seconds)

        self.fig = go.FigureWidget()

        self._init_axes()
        self._draw_calibration()

        self._init_lines()              # ECG traces FIRST
        self._init_overlay_traces()     # segments + noise
        self._init_legend_traces()      # legend entries
        self._init_fiducial_traces()    # fiducials (fast)
        self._init_info_annotations()   # info text

        self._init_static_view()

        self._init_widgets()            # display figure
        self._update_dispatch()         # first draw
        self._on_slider_widget({"new": 0})
    # ======================================================
    # AXES
    # ======================================================

    def _init_legend_traces(self):
        self._legend_traces = {}

        for k in list(self.labels.keys()):
            tr = go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=self.color_map[k], width=6),
                name=k,
                showlegend=True,
                visible=False,  # start hidden
            )
            self.fig.add_trace(tr)
            self._legend_traces[k] = tr

    def _init_fiducial_traces(self, max_fiducials=50):
        self._fiducial_traces = []

        for _ in range(max_fiducials):
            tr = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="gray", width=1),
                visible=False,
                showlegend=False,
            )
            self.fig.add_trace(tr)
            self._fiducial_traces.append(tr)
    def _init_info_annotations(self):
        self._info_ann_time = dict(
            x=0.5, y=0, xref="paper", yref="paper",
            text="",
            showarrow=False,
            font=dict(size=8),
        )
        self._info_ann_left = dict(
            x=0.08, y=0, xref="paper", yref="paper",
            text="",
            showarrow=False,
            font=dict(size=8),
        )
        self._info_ann_right = dict(
            x=0.81, y=0, xref="paper", yref="paper",
            text="",
            showarrow=False,
            font=dict(size=8),
        )

        anns = list(self.fig.layout.annotations or [])
        anns.extend([
            self._info_ann_left,
            self._info_ann_right,
            self._info_ann_time,
        ])
        self.fig.update_layout(annotations=anns)

    def _init_static_view(self):
        """
        Plotly equivalent of matplotlib static view + figure sizing.
        """

        # matplotlib default DPI ≈ 100
        DPI = 100

        width_in = 11.69 * (self.figure_samples / 5000)
        height_in = 8.27

        self.fig.update_layout(
            width=int(width_in * DPI),
            height=int(height_in * DPI),
        )
    def _init_axes(self):
        xmax = self.figure_samples * self.mm_per_sample
        ymax = self.num_channels * self.lead_spacing

        # ---- Axis configuration (no Plotly grid) ----
        self.fig.update_layout(
            xaxis=dict(
                range=[0, xmax],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=[-10, ymax],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            plot_bgcolor="white",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        self.fig.update_layout(
                legend=dict(
                    x=0.5,
                    y=1.02,
                    xanchor="center",
                    yanchor="bottom",
                    orientation="h",
                    traceorder="normal",   # 👈 ADD THIS
                )
            )

        # ---- Lead labels (same responsibility as matplotlib) ----
        annotations = []
        for ld, name in enumerate(self.lead_names):
            y = (self.num_channels - ld - 1) * self.lead_spacing
            annotations.append(
                        dict(
                            x=0,                 
                            xref="paper",       
                            y=y,
                            yref="y",
                            text=name,
                            showarrow=False,
                            xanchor="right",
                            yanchor="middle",
                            font=dict(size=12),
                        )
                    )

        # ---- ECG paper grid (STATIC) ----
        shapes = []

        # vertical grid lines (time)
        x = 0.0
        while x <= xmax:
            major = abs((x / 5) - round(x / 5)) < 1e-6
            shapes.append(
                dict(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=-10,
                    y1=ymax,
                    line=dict(
                        color="#ff5333",
                        width=0.6 if major else 0.2,
                    ),
                    layer="below",
                )
            )
            x += 1.0  # 1 mm

        # horizontal grid lines (amplitude)
        y = -10
        while y <= ymax:
            major = abs((y / 5) - round(y / 5)) < 1e-6
            shapes.append(
                dict(
                    type="line",
                    x0=0,
                    x1=xmax,
                    y0=y,
                    y1=y,
                    line=dict(
                        color="#ff5333",
                        width=0.6 if major else 0.2,
                    ),
                    layer="below",
                )
            )
            y += 1.0  # 1 mm

        self.fig.update_layout(
            annotations=annotations,
            shapes=shapes,
        )

    
    def _init_overlay_traces(self, max_segments=200, max_noise=200):
        self._segment_traces = []
        self._noise_traces = []

        # --- segments ---
        for _ in range(max_segments):
            tr = go.Scatter(
                x=[], y=[],
                mode="lines",
                fill="toself",
                fillcolor="rgba(0,0,0,0)",
                line=dict(width=0),
                visible=False,
                showlegend=False,
            )
            self.fig.add_trace(tr)
            self._segment_traces.append(tr)

        for _ in range(max_noise):
            tr = go.Scatter(
                x=[], y=[],
                mode="lines",
                fill="toself",
                fillcolor="rgba(0,0,0,0)",
                line=dict(width=0),
                visible=False,
                showlegend=False,
            )
            self.fig.add_trace(tr)
            self._noise_traces.append(tr)

    # ======================================================
    # LINES
    # ======================================================
    def _init_lines(self):
        for _ in range(self.num_channels):
            self.fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                          line=dict(color="lightgray", width=1),
                                          showlegend=False))
            self.fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                          line=dict(color="black", width=1.5),
                                          showlegend=False))
            self.fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                          line=dict(color="black", width=1),
                                          showlegend=False))
            
    def _draw_calibration(self):
        shapes = list(self.fig.layout.shapes or [])
        calib_height = self.gain
        calib_width = 5.0

        for ld in range(self.num_channels):
            center = (self.num_channels - ld - 1) * self.lead_spacing
            shapes.append(
                dict(
                    type="path",
                    path=(
                        f"M 0 {center} "
                        f"L 0 {center + calib_height} "
                        f"L {calib_width} {center + calib_height} "
                        f"L {calib_width} {center}"
                    ),
                    line=dict(color="#ff4c30", width=2),
                    layer="above",
                    name="calibration",
                )
            )

        self.fig.update_layout(shapes=shapes)

    # ======================================================
    # WIDGETS
    # ======================================================
    def _init_widgets(self):
        self.start_label = widgets.Label(self._fmt_dt(self.start_time))
        self.end_label = widgets.Label(self._fmt_dt(self.end_time))

        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.max_time,
            step=1,
            continuous_update=True,
            readout=False,
            layout=widgets.Layout(flex="1 1 auto"),
        )
        self.slider.observe(self._on_slider_widget, names="value")

        self.time_label = widgets.HTML()
        self.time_label_container = widgets.Box(
            [self.time_label],
            layout=widgets.Layout(width="100%", position="relative", height="20px"),
        )

        slider_row = widgets.HBox(
            [self.start_label, self.slider, self.end_label],
            layout=widgets.Layout(align_items="center"),
        )

        self.play_button = widgets.ToggleButton(
            value=False, description="▶ Play", icon="play"
        )
        self.segment_button = widgets.ToggleButton(
            value=True, description="Segments ON", icon="eye"
        )

        self.play_button.observe(self._on_play_toggle, names="value")
        self.segment_button.observe(self._on_segment_toggle, names="value")

        controls = widgets.HBox(
            [self.play_button, self.segment_button],
            layout=widgets.Layout(justify_content="center"),
        )

        display(
            widgets.VBox(
                [
                    self.fig,
                    self.time_label_container,
                    slider_row,
                    controls,
                ]
            )
        )

    # ======================================================
    # Formatting
    # ======================================================
    def _fmt_dt(self, t):
        return np.datetime_as_string(t, unit="s").replace("T", " ")

    # ======================================================
    # Slider
    # ======================================================
    def _on_slider_widget(self, change):
        t_sec = int(change["new"])
        current_time = self.start_time + np.timedelta64(t_sec, "s")

        frac = t_sec / self.max_time if self.max_time else 0.0
        left = min(max(int(frac * 100), 2), 98)

        self.time_label.value = (
            f"<div style='position:absolute; left:{left}%; "
            f"transform:translateX(-50%); font-size:12px;'>"
            f"{self._fmt_dt(current_time)}</div>"
        )

        self.current_sample = min(
            int(t_sec * self.fs),
            self.total_samples - self.window_samples,
        )
        self._update_dispatch()

    # ======================================================
    # Play / Pause
    # ======================================================
    def _on_play_toggle(self, change):
        self._playing = change["new"]

        if self._playing:
            self.play_button.description = "⏸ Pause"
            self.play_button.icon = "pause"
            self._start_playback()
        else:
            self.play_button.description = "▶ Play"
            self.play_button.icon = "play"

    def _start_playback(self):
        if self._play_thread and self._play_thread.is_alive():
            return

        def run():
            while self._playing:
                if self.slider.value >= self.slider.max:
                    self.play_button.value = False
                    break
                self.slider.value += 1
                time.sleep(self.play_interval)

        self._play_thread = threading.Thread(target=run, daemon=True)
        self._play_thread.start()

    # ======================================================
    # Segment toggle
    # ======================================================
    def _on_segment_toggle(self, change):
        self.show_segments = change["new"]
        self.segment_button.description = (
            "Segments ON" if self.show_segments else "Segments OFF"
        )
        self.segment_button.icon = "eye" if self.show_segments else "eye-slash"

        # legend needs full redraw
        self.legend = None
        self._init_static_view()
        self._update_dispatch()

    # ======================================================
    # UPDATE
    # ======================================================
    def _update_dispatch(self):
        self._update_fn()

    def _update(self):
        
        wf, corr, median, time_arr, cdf, ndf, fid = self._extract_window()

        n_ch, n_samp = wf.shape
        n_med = median.shape[1]

        # --------------------------------------------------
        # X coordinates
        # --------------------------------------------------
        x = np.arange(n_samp) * self.mm_per_sample
        mx = np.arange(n_med) * self.mm_per_sample + x[-1]

        # --------------------------------------------------
        # Y offsets
        # --------------------------------------------------
        y0 = (
            (self.num_channels - 1 - np.arange(self.num_channels))
            * self.lead_spacing
        ).reshape(-1, 1)

        y_corr = y0 + self.gain * corr
        y_wf   = y0 + self.gain * wf
        y_med  = y0 + self.gain * median

        # --------------------------------------------------
        # Trace updates (still looped, but now cheap)
        # --------------------------------------------------
        ctx = self.fig.batch_update()
        with ctx:
            ti = 0
            for ld in range(self.num_channels):
                self.fig.data[ti].x = x
                self.fig.data[ti].y = y_corr[ld]
                ti += 1

                self.fig.data[ti].x = x
                self.fig.data[ti].y = y_wf[ld]
                ti += 1

                self.fig.data[ti].x = mx
                self.fig.data[ti].y = y_med[ld]
                ti += 1

        # --------------------------------------------------
        # Overlays / annotations
        # --------------------------------------------------
        used = self._update_segments(cdf)
        has_noise = self._update_noise(ndf)
        self._update_legend(used, has_noise)
        self._update_fiducials(fid, mx)
        self._update_info_text(time_arr)

    # ======================================================
    # SEGMENTS
    # ======================================================
    def _update_segments(self, df):
        base_shapes = list(self.fig.layout.shapes or [])

        # remove old segment rectangles
        shapes = [
            s for s in base_shapes
            if not (s.type == "rect" and getattr(s, "name", None) == "segment")
        ]

        used = set()

        if not self.show_segments or df is None or df.empty:
            self.fig.update_layout(shapes=shapes)
            return used

        for _, row in df.iterrows():

            # ---- P wave ----
            if isinstance(row.get("p_type"), str) and "unspecified" not in row["p_type"]:
                label = row["p_type"]
                if label in self.labels:
                    shapes.append(dict(
                                    type="rect",
                                    xref="x",
                                    yref="y",
                                    x0=row["p_onset"] * self.mm_per_sample,
                                    x1=row["p_offset"] * self.mm_per_sample,
                                    y0=-self.half_height,
                                    y1=self.num_channels * self.lead_spacing,
                                    fillcolor=self.color_map[label],
                                    opacity=0.35,
                                    line_width=0,
                                    layer="above",
                                    name="segment",
                                ))

            # ---- QRS ----
            if isinstance(row.get("qrs_type"), str) and "unspecified" not in row["qrs_type"]:
                label = row["qrs_type"]
                if label in self.labels:
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=row["qrs_onset"] * self.mm_per_sample,
                        x1=row["qrs_offset"] * self.mm_per_sample,
                        y0=-self.half_height,
                        y1=self.num_channels * self.lead_spacing - 1,
                        fillcolor=self.color_map[label],
                        opacity=0.35,
                        line_width=0,
                        layer="above",
                        name="segment",
                    ))
                    used.add(label)

            # ---- T wave ----
            if "t_onset" in row and not pd.isna(row["t_onset"]):
                label = "t_wave"
                if label in self.labels:
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=row["t_onset"] * self.mm_per_sample,
                        x1=row["t_offset"] * self.mm_per_sample,
                        y0=-self.half_height,
                        y1=self.num_channels * self.lead_spacing - 1,
                        fillcolor=self.color_map[label],
                        opacity=0.35,
                        line_width=0,
                        layer="above",
                        name="segment",
                    ))
                    used.add(label)

        self.fig.update_layout(shapes=shapes)
        return used
    
    def _update_fiducials(self, fiducials, med_x):
        if fiducials is None:
            return

        # --- remove old fiducials ---
        shapes = [
            s for s in (self.fig.layout.shapes or [])
            if not (
                s.type == "line"
                and getattr(s, "name", None) == "fiducial"
            )
        ]

        idx = fiducials[~np.isnan(fiducials)].astype(int)
        idx = idx[(idx >= 0) & (idx < len(med_x))]

        for ld in range(self.num_channels):
            center = (self.num_channels - ld - 1) * self.lead_spacing

            for i in idx:
                x = med_x[i]
                shapes.append(
                    dict(
                        type="line",
                        x0=x,
                        x1=x,
                        y0=center - 5,
                        y1=center + 5,
                        line=dict(color="gray", width=1),
                        layer="above",
                        name="fiducial",
                    )
                )

        self.fig.update_layout(shapes=shapes)
    
    def _update_info_text(self, time_arr):
        # --- keep non-info annotations ---
        annotations = [
            a for a in (self.fig.layout.annotations or [])
            if getattr(a, "name", None) != "info"
        ]

        annotations.extend([
            dict(
                x=0.08, y=0, xref="paper", yref="paper",
                text=f"{self.window_samples/self.fs:.1f}s @ {self.fs} Hz",
                showarrow=False,
                font=dict(size=8),
                name="info",
            ),
            dict(
                x=0.81, y=0, xref="paper", yref="paper",
                text=f"{self.paper_speed} mm/s  {self.gain} mm/mV",
                showarrow=False,
                font=dict(size=8),
                name="info",
            ),
        ])

        if time_arr:
            t0 = str(time_arr[0]).split("T")[1].split(".")[0]
            t1 = str(time_arr[-1]).split("T")[1].split(".")[0]
            annotations.append(
                dict(
                    x=0.5, y=0, xref="paper", yref="paper",
                    text=f"{t0} → {t1}",
                    showarrow=False,
                    font=dict(size=8),
                    name="info",
                )
            )

        self.fig.update_layout(annotations=annotations)

    def _update_legend(self, used_types, has_noise):
        # hide everything first
        for tr in self._legend_traces.values():
            tr.visible = False

        # show used segment labels
        for t in used_types:
            if t in self._legend_traces:
                self._legend_traces[t].visible = True

        # show noise if needed
        if has_noise and "SEGMENTED_BEAT_TYPE_NOISE" in self._legend_traces:
            self._legend_traces["SEGMENTED_BEAT_TYPE_NOISE"].visible = True

    def _update_noise(self, df):
        base_shapes = list(self.fig.layout.shapes or [])

        # --- remove old NOISE rectangles only ---
        shapes = [
            s for s in base_shapes
            if not (s.type == "rect" and getattr(s, "name", None) == "SEGMENTED_BEAT_TYPE_NOISE")
        ]

        has_noise = False

        if df is None or df.empty:
            self.fig.update_layout(shapes=shapes)
            return has_noise

        for _, row in df.iterrows():
            if row.get("lead") not in self.lead_names:
                continue

            ld = self.lead_names.index(row["lead"])
            y = (self.num_channels - ld - 1) * self.lead_spacing

            shapes.append(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=row["onset"] * self.mm_per_sample,
                x1=row["offset"] * self.mm_per_sample,
                y0=y - self.half_height,
                y1=y + self.half_height,
                fillcolor=self.color_map["SEGMENTED_BEAT_TYPE_NOISE"],
                opacity=0.1,
                line_width=0,
                layer="above",
                name="SEGMENTED_BEAT_TYPE_NOISE",
            ))

            has_noise = True

        self.fig.update_layout(shapes=shapes)
        return has_noise
    
    def _update_fast(self):
        ctx = self.fig.batch_update()
        with ctx:
            wf, corr, median, time_arr, cdf, ndf, fid = self._extract_window()

            n_ch, n_samp = wf.shape
            n_med = median.shape[1]

            x = np.arange(n_samp) * self.mm_per_sample
            mx = np.arange(n_med) * self.mm_per_sample + x[-1]

            y0 = (
                (self.num_channels - 1 - np.arange(self.num_channels))
                * self.lead_spacing
            ).reshape(-1, 1)

            y_corr = y0 + self.gain * corr
            y_wf   = y0 + self.gain * wf
            y_med  = y0 + self.gain * median

            ti = 0
            for ld in range(self.num_channels):
                self.fig.data[ti].x = x
                self.fig.data[ti].y = y_corr[ld]; ti += 1
                self.fig.data[ti].x = x
                self.fig.data[ti].y = y_wf[ld]; ti += 1
                self.fig.data[ti].x = mx
                self.fig.data[ti].y = y_med[ld]; ti += 1

            used = self._update_segments_fast(cdf)
            has_noise = self._update_noise_fast(ndf)
            self._update_legend_fast(used, has_noise)
            self._update_fiducials_fast(fid, mx)
            self._update_info_text_fast(time_arr)
    
    def _update_segments_fast(self, df):
        used = set()

        # hide all
        for tr in self._segment_traces:
            tr.visible = False

        if not self.show_segments or df is None or df.empty:
            return used

        i = 0
        y0 = -self.half_height
        y1 = self.num_channels * self.lead_spacing

        for _, row in df.iterrows():
            if i >= len(self._segment_traces):
                break

            def draw(on, off, label):
                nonlocal i
                if label not in self.labels or pd.isna(on) or pd.isna(off):
                    return

                x0 = on * self.mm_per_sample
                x1 = off * self.mm_per_sample

                tr = self._segment_traces[i]
                tr.x = [x0, x1, x1, x0, x0]
                tr.y = [y0, y0, y1, y1, y0]
                tr.fillcolor = self.color_map[label]
                tr.visible = True

                used.add(label)
                i += 1

            if isinstance(row.get("p_type"), str) and "unspecified" not in row["p_type"]:
                draw(row["p_onset"], row["p_offset"], row["p_type"])

            if isinstance(row.get("qrs_type"), str) and "unspecified" not in row["qrs_type"]:
                draw(row["qrs_onset"], row["qrs_offset"], row["qrs_type"])

            if "t_onset" in row:
                draw(row["t_onset"], row["t_offset"], "t_wave")

        return used
    
    def _update_noise_fast(self, df):
        for tr in self._noise_traces:
            tr.visible = False

        if df is None or df.empty:
            return False

        i = 0
        has_noise = False

        for _, row in df.iterrows():
            if i >= len(self._noise_traces):
                break

            ld = self.lead_names.index(row["lead"]) if row.get("lead") in self.lead_names else None
            if ld is None:
                continue

            y = (self.num_channels - ld - 1) * self.lead_spacing
            x0 = row["onset"] * self.mm_per_sample
            x1 = row["offset"] * self.mm_per_sample

            tr = self._noise_traces[i]
            tr.x = [x0, x1, x1, x0, x0]
            tr.y = [
                y - self.half_height,
                y - self.half_height,
                y + self.half_height,
                y + self.half_height,
                y - self.half_height,
            ]
            tr.fillcolor = self.color_map["SEGMENTED_BEAT_TYPE_NOISE"]
            tr.visible = True
            has_noise = True
            i += 1

        return has_noise

    def _update_legend_fast(self, used, has_noise):
        for k, tr in self._legend_traces.items():
            tr.visible = (k in used) or (k == "SEGMENTED_BEAT_TYPE_NOISE" and has_noise)

    def _update_fiducials_fast(self, fiducials, med_x):
        # hide all first
        for tr in self._fiducial_traces:
            tr.visible = False

        if fiducials is None:
            return

        idx = fiducials[~np.isnan(fiducials)].astype(int)
        idx = idx[(idx >= 0) & (idx < len(med_x))]

        i = 0
        for ld in range(self.num_channels):
            center = (self.num_channels - ld - 1) * self.lead_spacing

            for j in idx:
                if i >= len(self._fiducial_traces):
                    return

                x = med_x[j]
                tr = self._fiducial_traces[i]

                tr.x = [x, x]
                tr.y = [center - 5, center + 5]
                tr.visible = True

                i += 1
    
    def _update_info_text_fast(self, time_arr):
        self._info_ann_left["text"] = (
            f"{self.window_samples/self.fs:.1f}s @ {self.fs} Hz"
        )
        self._info_ann_right["text"] = (
            f"{self.paper_speed} mm/s  {self.gain} mm/mV"
        )

        if time_arr:
            t0 = str(time_arr[0]).split("T")[1].split(".")[0]
            t1 = str(time_arr[-1]).split("T")[1].split(".")[0]
            self._info_ann_time["text"] = f"{t0} → {t1}"
        else:
            self._info_ann_time["text"] = ""

    # ======================================================
    # BUFFER (UNCHANGED LOGIC)
    # ======================================================
    def _assemble_buffer(self, center_seg):
        segments = []
        for s in range(center_seg - self.radius, center_seg + self.radius + 1):
            if 0 <= s < len(self.dataset):
                segments.append(select_case_from_sample(self.dataset.__getitem__(s)))

        wf, corr, times = [], [], []
        complex_dfs, noise_dfs = [], []

        SHIFT_COLS = {
            "onset",
            "offset",
            "p_onset",
            "p_offset",
            "qrs_onset",
            "qrs_offset",
            "t_onset",
            "t_offset",
        }

        for i, seg in enumerate(segments):
            wf.append(seg["waveform"])
            corr.append(seg["corrected_waveform"])
            times.extend(seg["time"])

            shift = i * self.seg_len

            # ==================================================
            # complex_df — SAME SEMANTICS AS BEFORE
            # ==================================================
            cdf = seg.get("complex_df")

            dfc = None
            if isinstance(cdf, pd.DataFrame):
                if not cdf.empty:
                    dfc = cdf.copy()

            elif isinstance(cdf, dict) and cdf:
                # dict-of-dicts → rows
                first_val = next(iter(cdf.values()))
                if isinstance(first_val, dict):
                    dfc = pd.DataFrame.from_records(list(cdf.values()))
                else:
                    # dict-of-lists → columns
                    dfc = pd.DataFrame(cdf)

            if dfc is not None and not dfc.empty:
                for c in SHIFT_COLS & set(dfc.columns):
                    dfc[c] = dfc[c] + shift
                complex_dfs.append(dfc)

            # ==================================================
            # noise_df — SAME AS ORIGINAL
            # ==================================================
            ndf = seg.get("noise_df")
            dfn = None

            if isinstance(ndf, pd.DataFrame):
                if not ndf.empty:
                    dfn = ndf.copy()

            elif isinstance(ndf, dict) and ndf:
                first_val = next(iter(ndf.values()))
                if isinstance(first_val, dict):
                    dfn = pd.DataFrame.from_records(list(ndf.values()))
                else:
                    dfn = pd.DataFrame(ndf)

            if dfn is not None and not dfn.empty:
                if "onset" in dfn:
                    dfn["onset"] += shift
                if "offset" in dfn:
                    dfn["offset"] += shift
                noise_dfs.append(dfn)

        return {
            "waveform": np.concatenate(wf, axis=1),
            "corrected": np.concatenate(corr, axis=1),
            "time": times,
            "complex_df": (
                pd.concat(complex_dfs, ignore_index=True)
                if complex_dfs
                else pd.DataFrame()
            ),
            "noise_df": (
                pd.concat(noise_dfs, ignore_index=True) if noise_dfs else pd.DataFrame()
            ),
            "median": segments[0]["median_beat"],
            "fiducials": segments[0]["fiducials"],
        }

    def _extract_window(self):
        center = self.current_sample // self.seg_len
        if center != self.loaded_center:
            self.buffer = self._assemble_buffer(center)
            self.loaded_center = center

        start = self.current_sample - (center - self.radius) * self.seg_len
        end = start + self.window_samples

        wf = self.buffer["waveform"][:, start:end]
        corr = self.buffer["corrected"][:, start:end]
        time = self.buffer["time"][start:end]
        median = self.buffer["median"]
        fid = self.buffer["fiducials"]

        cdf = self.buffer["complex_df"]
        ndf = self.buffer["noise_df"]

        if not cdf.empty:
            # main interval overlap (unchanged)
            cdf = cdf[(cdf["offset"] > start) & (cdf["onset"] < end)].copy()

            # shift everything
            for c in cdf.columns:
                if c.endswith("onset") or c.endswith("offset"):
                    cdf[c] -= start

            # invalidate sub-intervals that do NOT overlap the window
            def _invalidate(df, on, off):
                if on in df and off in df:
                    mask = (df[off] > 0) & (df[on] < self.window_samples)
                    df.loc[~mask, [on, off]] = np.nan

            _invalidate(cdf, "p_onset", "p_offset")
            _invalidate(cdf, "qrs_onset", "qrs_offset")
            _invalidate(cdf, "t_onset", "t_offset")

        if not ndf.empty:
            ndf = ndf[(ndf["offset"] > start) & (ndf["onset"] < end)].copy()
            ndf["onset"] -= start
            ndf["offset"] -= start

        return wf, corr, median, time, cdf, ndf, fid