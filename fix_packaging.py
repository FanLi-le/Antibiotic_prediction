#!/usr/bin/env python3
"""
快速修复 PyInstaller 打包问题的脚本。

这个脚本提供了几种不同的方法来修复 descriptastorus 和 RDKit 的打包问题。
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def method1_simple_fix():
    """方法1: 使用简单的命令行参数修复"""
    print("方法1: 使用简单修复参数打包...")
    
    cmd = [
        'pyinstaller',
        '--name=本草御菌',
        '--console',
        '--onedir',  # 使用 onedir 模式
        '--clean',
        # 收集所有必要的数据文件
        '--collect-data=descriptastorus',
        '--collect-data=rdkit',
        '--collect-data=chemprop',
      
        # 添加隐藏导入
        '--hidden-import=descriptastorus.descriptors.rdDescriptors',
        '--hidden-import=rdkit.Chem.Descriptors',
        '--hidden-import=chemprop',
        '--icon=app.ico',
        'app_beautiful_fixed_v2.py'
    ]
    
    print(f"执行: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0

def method2_with_hooks():
    """方法2: 使用自定义 hook 文件"""
    print("方法2: 使用自定义 hook 文件打包...")
    
    # 确保 hook 文件存在
    hook_dir = os.path.join(os.getcwd(), 'hooks')
    os.makedirs(hook_dir, exist_ok=True)
    
    # 复制 hook 文件到 hooks 目录
    if os.path.exists('hook-descriptastorus.py'):
        shutil.copy('hook-descriptastorus.py', os.path.join(hook_dir, 'hook-descriptastorus.py'))
    if os.path.exists('hook-rdkit.py'):
        shutil.copy('hook-rdkit.py', os.path.join(hook_dir, 'hook-rdkit.py'))
    
    cmd = [
        'pyinstaller',
        '--name=ChempropGUI',
        '--console',
        '--onedir',
        '--clean',
        f'--additional-hooks-dir={hook_dir}',
        '--collect-all=descriptastorus',
        '--collect-all=rdkit',
        '--collect-all=chemprop',
        'app_up.py'
    ]
    
    print(f"执行: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0

def method3_manual_data_files():
    """方法3: 手动指定数据文件路径"""
    print("方法3: 手动指定数据文件路径...")
    
    # 查找 Python 的 site-packages 路径
    import site
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else ''
    
    if not site_packages or not os.path.exists(site_packages):
        print("无法找到 site-packages 路径")
        return False
    
    cmd = [
        'pyinstaller',
        '--name=ChempropGUI',
        '--console',
        '--onedir',
        '--clean',
        # 手动添加数据文件
        f'--add-data={os.path.join(site_packages, "descriptastorus")}{os.pathsep}descriptastorus',
        f'--add-data={os.path.join(site_packages, "rdkit")}{os.pathsep}rdkit',
        # 隐藏导入
        '--hidden-import=descriptastorus.descriptors.rdDescriptors',
        '--hidden-import=rdkit.Chem.Descriptors',
        '--'
        'app_beautiful_fixed_v3.py'
    ]
    
    print(f"执行: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0

def method4_debug_mode():
    """方法4: 调试模式，生成详细的构建信息"""
    print("方法4: 调试模式打包...")
    
    cmd = [
        'pyinstaller',
        '--name=ChempropGUI',
        '--console',
        '--onedir',
        '--clean',
        '--debug=all',  # 启用所有调试信息
        '--log-level=DEBUG',
        '--collect-all=descriptastorus',
        '--collect-all=rdkit',
        'app_up.py'
    ]
    
    print(f"执行: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0

def main():
    """主函数"""
    
    print("Chemprop GUI 打包问题修复工具")
    print("=" * 50)
    
    # 检查 app_up.py 是否存在
    if not os.path.exists('app_up.py'):
        print("错误: 未找到 app_up.py 文件")
        return 1
    
    print("选择修复方法:")
    print("1. 简单修复 (推荐)")
    print("2. 使用自定义 hook 文件")
    print("3. 手动指定数据文件")
    print("4. 调试模式")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    methods = {
        '1': method1_simple_fix,
        '2': method2_with_hooks,
        '3': method3_manual_data_files,
        '4': method4_debug_mode,
    }
    
    if choice not in methods:
        print("无效的选择")
        return 1
    
    success = methods[choice]()
    
    if success:
        print("\\n✓ 打包完成！")
        print("输出目录: dist/ChempropGUI/")
        print("\\n重要提示:")
        print("1. 请测试生成的可执行文件是否能正常运行")
        print("2. 如果仍然有问题，请尝试其他方法")
        print("3. 方法4的调试信息可以帮助诊断问题")
    else:
        print("\\n✗ 打包失败！")
        print("请尝试其他方法或检查错误信息")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())