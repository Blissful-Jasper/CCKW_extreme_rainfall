#!/usr/bin/env python3
"""
通用气压层插值脚本
将3D变量（hus, ta, zg等）从模式层插值到标准气压层

特性:
- 支持有时间维度和无时间维度的数据
- 支持断点续传
- 内存高效处理
- 自动处理 cntl, p4k, 4co2 三个实验

作者: Based on 63_divergence_interp_pressure.ipynb
日期: 2026-02-17
"""

import xarray as xr
import numpy as np
from pathlib import Path
import argparse
import sys


def hacky_plvl_interpolation_plus(data_var, data_pfull, plvl_target, unit='hPa', 
                                  label='pressure level', height_name='level'):
    """
    将变量插值到特定气压层
    
    Parameters
    ----------
    data_var : xr.DataArray
        待插值的变量
    data_pfull : xr.DataArray
        气压场（单位与 plvl_target 相同）
    plvl_target : float
        目标气压层
    unit : str
        气压层单位
    label : str
        气压层长名称
    height_name : str
        识别垂直维度的子字符串
        
    Returns
    -------
    xr.DataArray
        插值后的变量在目标气压层的值
    """
    # 更稳健的维度查找
    var_height_dim = next((dim for dim in data_var.dims if height_name in dim), None)
    pfull_height_dim = next((dim for dim in data_pfull.dims if height_name in dim), None)
    
    if var_height_dim is None or pfull_height_dim is None:
        raise ValueError(f"无法找到包含 '{height_name}' 的维度")
    
    # 查找插值所需的层级（必须 compute 才能用作索引器）
    level_above = (data_pfull > plvl_target).argmax(dim=pfull_height_dim).compute()
    level_below = level_above - 1
    
    # 获取相邻层的值
    value_above = data_pfull.isel({pfull_height_dim: level_above})
    value_below = data_pfull.isel({pfull_height_dim: level_below})
    
    # 线性插值权重
    f = (plvl_target - value_below) / (value_above - value_below)
    
    # 插值变量
    data_interpolated = (
        (1 - f) * data_var.isel({var_height_dim: level_below}) + 
        f * data_var.isel({var_height_dim: level_above})
    )
    
    # 清理并添加维度
    data_interpolated = (
        data_interpolated
        .drop_vars(pfull_height_dim, errors='ignore')
        .expand_dims(dim={"plev": [plvl_target]}, axis=0)
    )
    
    # 添加元数据
    data_interpolated['plev'].attrs = {
        'standard_name': 'air_pressure',
        'long_name': label,
        'units': unit,
        'axis': 'Z',
        'positive': 'down'
    }
    
    return data_interpolated


def interpolate_to_pressure_levels(data_var, data_pfull, plvl_list, 
                                   pressure_unit_conversion=100, 
                                   height_name='level',
                                   compute_each_level=False):
    """
    高效地将变量插值到多个气压层
    
    Parameters
    ----------
    data_var : xr.DataArray
        待插值的变量
    data_pfull : xr.DataArray
        气压场
    plvl_list : list of float
        目标气压层（hPa）
    pressure_unit_conversion : float
        将 data_pfull 转换为 hPa 的因子（默认: 100，Pa->hPa）
    height_name : str
        识别垂直维度的子字符串
    compute_each_level : bool
        如果为 True，逐层计算（内存受限系统使用）
        
    Returns
    -------
    xr.DataArray
        所有目标气压层的插值变量
    """
    # 如果需要，将气压转换为 hPa
    if pressure_unit_conversion != 1:
        data_pfull = data_pfull / pressure_unit_conversion
    
    if compute_each_level:
        # 内存高效：逐层计算
        level_results = []
        for i, plvl in enumerate(plvl_list):
            print(f"    [{i+1}/{len(plvl_list)}] {plvl} hPa", flush=True)
            level_data = hacky_plvl_interpolation_plus(
                data_var, data_pfull, plvl, height_name=height_name
            ).compute()  # 使用 Dask 计算
            level_results.append(level_data)
        result = xr.concat(level_results, dim='plev')
    else:
        # 更快：构建计算图然后一次计算（推荐，充分利用 Dask 并行）
        print(f"    构建计算图 ({len(plvl_list)} 个气压层)...", flush=True)
        interpolated_levels = [
            hacky_plvl_interpolation_plus(data_var, data_pfull, plvl, height_name=height_name)
            for plvl in plvl_list
        ]
        result = xr.concat(interpolated_levels, dim='plev')
        print(f"    🚀 Dask 并行计算所有气压层...", flush=True)
        result = result.compute()  # Dask 会自动并行执行所有层的插值
    
    return result


def interpolate_variable_to_pressure(
    var_name,
    experiments=['cntl', 'p4k', '4co2'],
    plvl_list=[50, 100, 200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 1000],
    data_dir='/work/mh1498/m301257/3D_data',
    output_dir=None,
    lat_range=None,
    pressure_var='pfull',
    compute_each_level=False,
    resume=True,
    has_time_dim=True
):
    """
    将变量从模式层插值到气压层
    
    Parameters
    ----------
    var_name : str
        变量名称（如 'hus', 'ta', 'zg'）
    experiments : list of str
        实验名称列表
    plvl_list : list of float
        目标气压层（hPa）
    data_dir : str or Path
        数据目录
    output_dir : str or Path, optional
        输出目录（默认: data_dir/pressure_levels）
    lat_range : tuple, optional
        纬度范围 (lat_min, lat_max)
    pressure_var : str
        气压变量名称（'pfull' 或 'phalf'）
    compute_each_level : bool
        逐层计算（省内存）
    resume : bool
        支持断点续传
    has_time_dim : bool
        数据是否有时间维度
        
    Returns
    -------
    dict
        包含各实验插值结果的字典
    """
    data_dir = Path(data_dir)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = data_dir / 'pressure_levels'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  变量插值到气压层: {var_name}")
    print(f"{'='*70}")
    print(f"目标气压层: {plvl_list} hPa")
    print(f"实验: {experiments}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    if lat_range:
        print(f"纬度范围: {lat_range}")
    print(f"气压变量: {pressure_var}")
    print(f"计算模式: {'逐层计算（省内存）' if compute_each_level else '批量计算（更快）'}")
    print(f"断点续传: {'启用' if resume else '禁用'}")
    print(f"{'='*70}\n")
    
    results = {}
    
    for exp in experiments:
        output_file = output_dir / f'{var_name}_pressure_levels_{exp}.nc'
        
        # 检查是否已存在（断点续传）
        if resume and output_file.exists():
            print(f"✓ {exp.upper()}: 文件已存在，跳过: {output_file}", flush=True)
            # 尝试加载为 DataArray，如果失败则从 Dataset 中提取
            try:
                results[exp] = xr.open_dataarray(output_file)
            except ValueError:
                ds = xr.open_dataset(output_file)
                # 查找变量名对应的数据变量
                if var_name in ds.data_vars:
                    results[exp] = ds[var_name]
                else:
                    # 如果找不到，使用第一个数据变量
                    first_var = list(ds.data_vars)[0]
                    results[exp] = ds[first_var]
                    print(f"  注意: 使用变量 '{first_var}' 而非 '{var_name}'")
            continue
        
        print(f"\n{'='*70}")
        print(f"  处理 {exp.upper()}")
        print(f"{'='*70}")
        
        # 加载数据
        var_file = data_dir / exp / f'{var_name}_all_levels.nc'
        pressure_file = data_dir / exp / f'{pressure_var}_all_levels.nc'
        
        if not var_file.exists():
            print(f"⚠ 警告: 文件不存在，跳过: {var_file}")
            continue
        if not pressure_file.exists():
            print(f"⚠ 警告: 文件不存在，跳过: {pressure_file}")
            continue
        
        print(f"  加载变量: {var_file}")
        print(f"  加载气压: {pressure_file}")
        
        # 使用 chunks 加载以提高性能
        # 智能分块：根据数据大小自动调整
        if has_time_dim:
            chunks = {'time': 1000, 'lat': 'auto', 'lon': 'auto', 'level': -1}
        else:
            chunks = {'lat': 'auto', 'lon': 'auto', 'level': -1}
        
        var_data = xr.open_dataset(var_file, chunks=chunks)
        pressure_data = xr.open_dataset(pressure_file, chunks=chunks)
        
        # 选择纬度范围
        if lat_range is not None:
            var_data = var_data.sel(lat=slice(lat_range[0], lat_range[1]))
            pressure_data = pressure_data.sel(lat=slice(lat_range[0], lat_range[1]))
            print(f"  纬度范围: [{lat_range[0]}, {lat_range[1]}]")
            
            # 对齐纬度坐标 - 如果两者的纬度点不完全相同
            if not var_data['lat'].equals(pressure_data['lat']):
                print(f"  ⚠ 检测到纬度坐标不一致，对齐到气压场的纬度...")
                print(f"    变量纬度点数: {len(var_data['lat'].values)}")
                print(f"    气压纬度点数: {len(pressure_data['lat'].values)}")
                
                # 将变量数据对齐到气压场的纬度
                var_data = var_data.interp(lat=pressure_data['lat'], method='linear')
                print(f"    对齐后变量纬度点数: {len(var_data['lat'].values)}")
        
        print(f"  变量形状: {var_data[var_name].shape}")
        print(f"  气压形状: {pressure_data[pressure_var].shape}")
        
        # 插值
        print(f"  开始插值...")
        result = interpolate_to_pressure_levels(
            var_data[var_name],
            pressure_data[pressure_var],
            plvl_list,
            pressure_unit_conversion=100,  # Pa -> hPa
            height_name='level',
            compute_each_level=compute_each_level
        )
        
        # 保存
        print(f"  保存到: {output_file}")
        result.to_netcdf(output_file)
        print(f"✓ 完成 {exp.upper()}")
        
        results[exp] = result
    
    print(f"\n{'='*70}")
    print(f"  所有处理完成！")
    print(f"{'='*70}")
    print(f"输出文件:")
    for exp in experiments:
        output_file = output_dir / f'{var_name}_pressure_levels_{exp}.nc'
        if output_file.exists():
            print(f"  ✓ {output_file}")
    print(f"{'='*70}\n")
    
    return results


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description='将3D变量从模式层插值到标准气压层',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 插值 hus 和 ta（有时间维度）
  python interpolate_to_pressure_levels.py --var hus ta --lat-range -16 16
  
  # 插值 zg（无时间维度）
  python interpolate_to_pressure_levels.py --var zg --no-time-dim
  
  # 自定义气压层
  python interpolate_to_pressure_levels.py --var hus --plev 100 250 500 850
  
  # 省内存模式
  python interpolate_to_pressure_levels.py --var hus --compute-each-level
        """
    )
    
    parser.add_argument('--var', nargs='+', required=True,
                       help='变量名称（如 hus ta zg）')
    parser.add_argument('--experiments', nargs='+', default=['cntl', 'p4k', '4co2'],
                       help='实验名称（默认: cntl p4k 4co2）')
    parser.add_argument('--plev', nargs='+', type=float,
                       default=[50, 100, 200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 1000],
                       help='目标气压层（hPa）')
    parser.add_argument('--data-dir', default='/work/mh1498/m301257/3D_data',
                       help='数据目录')
    parser.add_argument('--output-dir', default=None,
                       help='输出目录（默认: data_dir/pressure_levels）')
    parser.add_argument('--lat-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='纬度范围，如: --lat-range -16 16')
    parser.add_argument('--pressure-var', default='pfull', choices=['pfull', 'phalf'],
                       help='气压变量名称（默认: pfull）')
    parser.add_argument('--compute-each-level', action='store_true',
                       help='逐层计算（省内存但较慢）')
    parser.add_argument('--no-resume', action='store_true',
                       help='禁用断点续传')
    parser.add_argument('--no-time-dim', action='store_true',
                       help='数据无时间维度（如 zg）')
    
    args = parser.parse_args()
    
    # 处理每个变量
    for var_name in args.var:
        interpolate_variable_to_pressure(
            var_name=var_name,
            experiments=args.experiments,
            plvl_list=args.plev,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            lat_range=tuple(args.lat_range) if args.lat_range else None,
            pressure_var=args.pressure_var,
            compute_each_level=args.compute_each_level,
            resume=not args.no_resume,
            has_time_dim=not args.no_time_dim
        )


if __name__ == '__main__':
    main()
