from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from typing import Optional, Union
import warnings
import geocat.viz.util as gvutil
import cmaps

# plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "regular"

OUT_DIR = Path("./figures/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Broader western-Pacific domain. Longitudes are in degrees east.
WESTERN_PACIFIC_EXTENT = [100, 250, -30, 50]

print(f"Figure output directory: {OUT_DIR}")

def _turn_off_cartopy_ticks(ax):
    ax.tick_params(labelsize=15, direction='out', top=False, right=False)
    # 1. 关掉投影轮廓
    try:
        ax.spines['geo'].set_visible(False)
    except KeyError:
        ax.outline_patch.set_visible(False)

    from matplotlib.lines import Line2D

    # 关闭 geo 轮廓
    try:
        ax.spines['geo'].set_visible(False)
    except KeyError:
        ax.outline_patch.set_visible(False)
    plt.draw()  # 先触发渲染，让 cartopy 生成所有 tick

    # 关闭顶部和右侧的 tick 线
    ax.xaxis.set_tick_params(top=False, which='both')
    ax.yaxis.set_tick_params(right=False, which='both')
    # 左边线
    ax.add_artist(Line2D([0, 0], [0, 1], transform=ax.transAxes,
                        color='black', linewidth=1.5, clip_on=False))
    # 底边线
    ax.add_artist(Line2D([0, 1], [0, 0], transform=ax.transAxes,
                        color='black', linewidth=1.5, clip_on=False))
    

def _format_standard_lon_lat_ticks(ax, extent, lon_step, lat_step):
    xticks = np.arange(extent[0], extent[1] + 1, lon_step)
    yticks = np.arange(extent[2], extent[3] + 1, lat_step)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True, dateline_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(which="major", direction="in", length=4, width=1, pad=8, labelsize=11)


def _set_geocat_lon_lat_ticks(ax, extent, central_lon, lon_step, lat_step):
    import geocat.viz as gv

    lon_range = extent[0:2]
    lat_range = extent[2:4]
    if central_lon == 180:
        xlim = (lon_range[0] - 180, lon_range[1] - 180)
        xticks_vals = np.arange(lon_range[0], lon_range[1] + 1, lon_step) - 180
    else:
        xlim = lon_range
        xticks_vals = np.arange(lon_range[0], lon_range[1] + 1, lon_step)

    gv.set_axes_limits_and_ticks(
        ax,
        xlim=xlim,
        ylim=lat_range,
        xticks=xticks_vals,
        yticks=np.arange(lat_range[0], lat_range[1] + 1, lat_step),
    )
    gv.add_major_minor_ticks(ax, labelsize=11)
    gv.add_lat_lon_ticklabels(ax)
    ax.tick_params(labeltop=False, labelright=False)


def plot_western_pacific_basemap(
    extent=WESTERN_PACIFIC_EXTENT,
    central_lon=180,
    figsize=(14, 7.5),
    title="Western Pacific",
    use_geocat=True,
    show_labels=True,
    add_reference_lines=True,
    land_facecolor="lightgray",
    coastline_color="gray",
    transparent=False,
    save=True,
    basename="western_pacific_basemap_provided_style",
    fig=None, ax=None,
):
    """Draw a Western Pacific basemap following the provided Cartopy/GeoCAT style."""
    if use_geocat:
        try:
            import geocat.viz as gv
        except ImportError:
            print("geocat.viz is not installed; using standard Cartopy axis setup")
            use_geocat = False

    lon_step = 60 if (extent[1] - extent[0]) > 270 else 20
    lat_step = 30 if (extent[3] - extent[2]) > 60 else 15

    if fig is None or ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_lon)},
        )

    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    if use_geocat:
        _set_geocat_lon_lat_ticks(ax, extent, central_lon, lon_step, lat_step)
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        _format_standard_lon_lat_ticks(ax, extent, lon_step, lat_step)
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

    if not show_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    if land_facecolor is not None:
        ax.add_feature(cfeature.LAND, facecolor=land_facecolor, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor=coastline_color, linewidth=0.5, zorder=1)
    ax.coastlines(linewidth=0.5, alpha=0.6)

    if add_reference_lines:
        ax.plot(
            [extent[0], extent[1]],
            [0, 0],
            color="gray",
            linewidth=0.8,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        ax.plot(
            [180, 180],
            [extent[2], extent[3]],
            color="gray",
            linewidth=0.8,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    if title:
        if use_geocat:
            gv.set_titles_and_labels(ax, maintitle=title, maintitlefontsize=13)
        else:
            ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=8)

    if save:
        png_path = OUT_DIR / f"{basename}.png"
        pdf_path = OUT_DIR / f"{basename}.pdf"
        svg_path = OUT_DIR / f"{basename}.svg"
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", transparent=transparent)
        fig.savefig(pdf_path, format="pdf", dpi=600, bbox_inches="tight", transparent=transparent)
        fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=transparent)
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")
        print(f"Saved {svg_path}")

    return fig, ax




# ── AGU 规范常量 ──────────────────────────────────────────────────────────────
AGU_SINGLE_COL  = 3.74   # inches,  ~95 mm
AGU_DOUBLE_COL  = 7.48   # inches, ~190 mm
AGU_MAX_HEIGHT  = 9.0    # inches, ~228 mm
AGU_FONT_FAMILY = 'sans-serif'
AGU_FONT_SIZE   = {
    'title'      : 8,
    'label'      : 8,
    'tick'       : 7,
    'clabel'     : 7,
    'cbar_label' : 7,
    'legend'     : 7,
}


def _auto_levels(data: np.ndarray,
                 n_levels: int = 21,
                 symmetric: bool = False,
                 percentile_clip: float = 2.0
                 ) -> np.ndarray:
    """
    根据数据自动计算 contour levels。
    - symmetric=True  → 以 0 为中心（适合异常场）
    - percentile_clip → 裁掉两端极值，避免 outlier 撑坏色阶
    """
    lo = np.nanpercentile(data, percentile_clip)
    hi = np.nanpercentile(data, 100 - percentile_clip)

    if symmetric:
        bound = max(abs(lo), abs(hi))
        # 取整到"好看"的数
        bound = _nice_number(bound)
        return np.linspace(-bound, bound, n_levels)
    else:
        lo = _nice_number(lo, round_down=True)
        hi = _nice_number(hi, round_down=False)
        return np.linspace(lo, hi, n_levels)


def _nice_number(x: float, round_down: bool = False) -> float:
    """把数字圆整到 1/2/5 系列，方便刻度对齐。"""
    if x == 0:
        return 0.0
    sign  = np.sign(x)
    x_abs = abs(x)
    exp   = np.floor(np.log10(x_abs))
    frac  = x_abs / 10**exp
    if round_down:
        nice = 1 if frac < 2 else (2 if frac < 5 else 5)
    else:
        nice = 2 if frac <= 1 else (5 if frac <= 2 else 10)
    return sign * nice * 10**exp


def _auto_figsize(col_width: str = 'single',
                  aspect: float = 0.75) -> tuple:
    """
    col_width : 'single' | 'double' | 'full'
    aspect    : height / width
    """
    w = {'single': AGU_SINGLE_COL,
         'double': AGU_DOUBLE_COL,
         'full'  : AGU_DOUBLE_COL}[col_width]
    h = min(w * aspect, AGU_MAX_HEIGHT)
    return (w, h)


def _apply_agu_style(ax,
                     xlabel: str = '',
                     ylabel: str = '',
                     title : str = '',
                     title_loc: str = 'left'):
    """统一应用 AGU 字体、轴线宽度、tick 方向等规范。"""
    # 标题与标签
    ax.set_title(title, loc=title_loc,
                 fontsize=AGU_FONT_SIZE['title'], fontweight='bold', pad=3)
    ax.set_xlabel(xlabel, fontsize=AGU_FONT_SIZE['label'], labelpad=3)
    ax.set_ylabel(ylabel, fontsize=AGU_FONT_SIZE['label'], labelpad=3)

    # tick 规范
    ax.tick_params(axis='both', which='major',
                   labelsize=AGU_FONT_SIZE['tick'],
                   direction='in', length=3, width=0.6,
                   top=False, right=False)
    ax.tick_params(axis='both', which='minor',
                   direction='in', length=1.5, width=0.4,
                   top=False, right=False)

    # 轴线宽度
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def set_ax_axis_major_interval(ax):
    ax.xaxis.set_major_locator(mticker.MultipleLocator(90))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(3))
    ax.xaxis.set_major_formatter(LongitudeFormatter())





# ── 常量 ─────────────────────────────────────────────────────────────────────
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']


# ── 轴格式化工具 ──────────────────────────────────────────────────────────────
def makefig_lon(ax, title, box):
    """经度轴格式：x 轴为 longitude。"""
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.set_xlim(box[0], box[1])
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    ax.tick_params(which='major', length=4, width=0.6, direction='in', labelsize=8)
    ax.tick_params(which='minor', length=2, width=0.4, direction='in')
    return ax


def makefig_lat(ax, title, box):
    """纬度轴格式：x 轴为 latitude。"""
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.xaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xlim(box[2], box[3])
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    ax.tick_params(which='major', length=4, width=0.6, direction='in', labelsize=8)
    ax.tick_params(which='minor', length=2, width=0.4, direction='in')
    return ax


def _add_colorbar(fig, ax, cf, label='(mm/day)²',
                  ticks=None, orientation='vertical'):
    """统一 colorbar 样式。"""
    cbar = fig.colorbar(cf, ax=ax,
                        orientation=orientation,
                        shrink=0.95, aspect=28, pad=0.03,
                        ticks=ticks)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(which='both', direction='in',
                        length=3, width=0.5, labelsize=7)
    cbar.outline.set_linewidth(0.5)
    return cbar


def _base_plot(ax, x, time, data, cmap, levels, contour_colors='k'):
    """底层绘图：contourf + contour + grid，返回 cf, cs。"""
    cf = ax.contourf(x, time, data,
                     cmap=cmap,
                     levels=levels,
                     extend='both')
    cs = ax.contour(x, time, data,
                    colors=contour_colors,
                    linewidths=0.6,
                    alpha=0.8,
                    levels=levels[::max(1, len(levels)//8)])
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f',
              colors=contour_colors, use_clabeltext=True)
    ax.grid(linestyle='--', color='gray', alpha=0.3, linewidth=0.5)
    return cf, cs


def _set_yaxis_month(ax, y_ticks=None, y_labels=None):
    """统一月份 y 轴。"""
    y_ticks  = y_ticks  if y_ticks  is not None else np.arange(1, 13)
    y_labels = y_labels if y_labels is not None else MONTH_LABELS
    ax.set_ylim(1, 12)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylabel('Month', fontsize=8, labelpad=3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis='y', which='both', direction='in',
                   length=3, width=0.5, right=False)


# ── 纬向平均（x 轴为纬度）演变图 ─────────────────────────────────────────────
def plot_data_lat(data, ax, fig,
                  lat,
                  time,
                  cmap=None,
                  levels=21,
                  title='',
                  box=(0, 360, -20, 20),
                  y_ticks=None,
                  y_labels=None,
                  cbar_label='(mm/day)²',
                  cbar_ticks=None,
                  add_cbar=True):
    """
    纬度–时间 Hovmöller（纬向平均演变）。

    Parameters
    ----------
    data       : 2-D array, shape (time, lat)
    ax, fig    : matplotlib Axes / Figure
    lat        : 1-D latitude array
    time       : 1-D time/month array (1–12)
    levels     : int 或 array，传 int 时自动计算
    box        : [lon_min, lon_max, lat_min, lat_max]
    """
    cmap   = cmap or cmaps.BlueWhiteOrangeRed
    levels = (np.linspace(np.nanpercentile(data, 2),
                          np.nanpercentile(data, 98), 21)
              if isinstance(levels, int) else levels)

    cf, _ = _base_plot(ax, lat, time, data, cmap, levels)
    _set_yaxis_month(ax, y_ticks, y_labels)
    makefig_lat(ax, title, box)

    if add_cbar:
        _add_colorbar(fig, ax, cf, label=cbar_label, ticks=cbar_ticks)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)

    return cf, ax


# ── 经向平均（x 轴为经度）演变图 ─────────────────────────────────────────────
def plot_data_lon(data, ax, fig,
                  lon,
                  time,
                  cmap=None,
                  levels=21,
                  title='',
                  box=(0, 360, -20, 20),
                  y_ticks=None,
                  y_labels=None,
                  cbar_label='(mm/day)²',
                  cbar_ticks=None,
                  add_cbar=True):
    """
    经度–时间 Hovmöller（经向平均演变）。

    Parameters
    ----------
    data       : 2-D array, shape (time, lon)
    ax, fig    : matplotlib Axes / Figure
    lon        : 1-D longitude array
    time       : 1-D time/month array (1–12)
    levels     : int 或 array，传 int 时自动计算
    box        : [lon_min, lon_max, lat_min, lat_max]
    """
    cmap   = cmap or cmaps.WhiteBlueGreenYellowRed
    levels = (np.linspace(np.nanpercentile(data, 2),
                          np.nanpercentile(data, 98), 21)
              if isinstance(levels, int) else levels)

    cf, _ = _base_plot(ax, lon, time, data, cmap, levels)
    _set_yaxis_month(ax, y_ticks, y_labels)
    makefig_lon(ax, title, box)

    if add_cbar:
        _add_colorbar(fig, ax, cf, label=cbar_label, ticks=cbar_ticks)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)

    return cf, ax









def plot_hovmoller(
    lon       : np.ndarray,
    time_coord: np.ndarray,
    data      : np.ndarray,
    *,
    # ── 数据范围 ──
    levels        : Optional[np.ndarray] = None,
    n_levels      : int   = 21,
    symmetric     : bool  = False,
    percentile_clip: float = 2.0,
    # ── 颜色 ──
    cmap          : str   = 'RdBu_r',
    contour_color : str   = 'k',
    contour_lw    : float = 0.6,
    contour_alpha : float = 0.8,
    add_contour   : bool  = True,
    contour_fmt   : str   = '%.1f',
    # ── 月份轴 ──
    yticks        : Optional[np.ndarray] = None,
    yticklabels   : Optional[list]       = None,
    ylim          : Optional[tuple]      = None,
    # ── 经度轴 ──
    xtick_spacing : float = 30.0,
    # ── 图形 ──
    col_width     : str   = 'double',
    aspect        : float = 0.65,
    figsize       : Optional[tuple] = None,
    # ── 文字 ──
    title         : str   = '',
    title_loc     : str   = 'left',
    xlabel        : str   = 'Longitude (°E)',
    ylabel        : str   = 'Month',
    cbar_label    : str   = '',
    # ── 输出 ──
    save_path     : Optional[str] = None,
    dpi           : int   = 300,
    ax            : Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    通用 Hovmöller（经度–时间）绘图函数，符合 AGU 投稿规范。

    Parameters
    ----------
    lon        : 1-D longitude array
    time_coord : 1-D time/month array (numeric, e.g. 1–12)
    data       : 2-D array, shape (len(time_coord), len(lon))
    levels     : 若指定则直接使用，否则自动计算
    symmetric  : True → 异常场，色阶关于 0 对称
    col_width  : 'single'(3.74") | 'double'(7.48")
    save_path  : 若指定则保存为文件（pdf/png 均可）

    Returns
    -------
    fig, ax
    """
    # ── 0. 自动 levels ────────────────────────────────────────────────────────
    if levels is None:
        levels = _auto_levels(data, n_levels=n_levels,
                              symmetric=symmetric,
                              percentile_clip=percentile_clip)

    # ── 1. 创建画布 ───────────────────────────────────────────────────────────
    if ax is None:
        fs = figsize or _auto_figsize(col_width, aspect)
        fig, ax = plt.subplots(figsize=fs)
    else:
        fig = ax.get_figure()

    # ── 2. 填色 contourf ──────────────────────────────────────────────────────
    cf = ax.contourf(lon, time_coord, data,
                     levels=levels, cmap=cmap,
                     extend='both')

    # ── 3. 等值线 contour ─────────────────────────────────────────────────────
    if add_contour:
        # 只画整数或"好看"的 levels
        cl_levels = levels[::max(1, len(levels)//10)]  # ~10条等值线
        cs = ax.contour(lon, time_coord, data,
                        levels=cl_levels,
                        colors=contour_color,
                        linewidths=contour_lw,
                        alpha=contour_alpha,
                        linestyles='-')
        ax.clabel(cs, inline=True,
                  fontsize=AGU_FONT_SIZE['clabel'],
                  fmt=contour_fmt,
                  colors=contour_color,
                  use_clabeltext=True)

    # ── 4. 月份 y 轴 ──────────────────────────────────────────────────────────
    DEFAULT_MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec']
    if yticks is None and ylim is None:
        # 自动判断是否是月份轴
        if len(time_coord) <= 12 and time_coord.max() <= 12:
            yticks      = np.arange(1, 13)
            yticklabels = yticklabels or DEFAULT_MONTH_LABELS
            ylim        = (1, 12)

    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=AGU_FONT_SIZE['tick'])
    if ylim is not None:
        ax.set_ylim(ylim)

    # ── 5. 经度 x 轴 ──────────────────────────────────────────────────────────
    lon_min, lon_max = float(lon.min()), float(lon.max())
    xticks = np.arange(
        np.ceil(lon_min / xtick_spacing) * xtick_spacing,
        lon_max + 1,
        xtick_spacing
    )
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{int(v)}°E' if v >= 0 else f'{int(-v)}°W')
    )
    ax.set_xlim(lon_min, lon_max)

    # ── 6. AGU 样式 ───────────────────────────────────────────────────────────
    _apply_agu_style(ax, xlabel=xlabel, ylabel=ylabel,
                     title=title, title_loc=title_loc)

    # ── 7. Colorbar ───────────────────────────────────────────────────────────
    cbar = fig.colorbar(cf, ax=ax,
                        orientation='vertical',
                        pad=0.02, shrink=0.95, aspect=25,
                        extend='both')
    cbar.ax.tick_params(labelsize=AGU_FONT_SIZE['cbar_label'],
                        direction='in', length=2, width=0.5)
    cbar.set_label(cbar_label, fontsize=AGU_FONT_SIZE['cbar_label'])
    cbar.outline.set_linewidth(0.5)

    # ── 8. 保存 ───────────────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, dpi=dpi,
                    bbox_inches='tight', facecolor='white',
                    metadata={'Creator': 'plot_hovmoller'})
        print(f'Saved → {save_path}')

    fig.tight_layout()
    return fig, ax
