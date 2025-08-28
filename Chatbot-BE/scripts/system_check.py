#!/usr/bin/env python3
"""
System Resource Check for RAG Chatbot

This script checks if your system has adequate resources to run the RAG chatbot
with Ollama, sentence-transformers, and FAISS.
"""

import psutil
import platform
import os
import sys
from pathlib import Path

def check_system_resources():
    """Check system resources and provide recommendations."""
    
    print("🔍 SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    # CPU Information
    print(f"💻 CPU: {platform.processor()}")
    print(f"   Cores: {psutil.cpu_count()} (Physical: {psutil.cpu_count(logical=False)})")
    print(f"   Architecture: {platform.architecture()[0]}")
    
    # Memory Information
    memory = psutil.virtual_memory()
    print(f"\n🧠 MEMORY:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    print(f"   Used: {memory.percent}%")
    
    # Disk Information
    disk = psutil.disk_usage('/')
    print(f"\n💾 DISK:")
    print(f"   Total: {disk.total / (1024**3):.1f} GB")
    print(f"   Free: {disk.free / (1024**3):.1f} GB")
    print(f"   Used: {disk.percent}%")
    
    # Check available disk space in project directory
    project_dir = Path(__file__).parent.parent
    try:
        project_disk = psutil.disk_usage(str(project_dir))
        print(f"   Project directory free: {project_disk.free / (1024**3):.1f} GB")
    except:
        print("   Project directory free: Unable to check")
    
    # Python and Package Information
    print(f"\n🐍 PYTHON:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    
    # Check for required packages
    print(f"\n📦 PACKAGE CHECK:")
    required_packages = [
        'sentence_transformers',
        'faiss',
        'numpy',
        'requests',
        'pypdf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    # Ollama Check
    print(f"\n🤖 OLLAMA CHECK:")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama server running")
            print(f"   Available models: {len(models)}")
            for model in models[:5]:  # Show first 5 models
                print(f"     - {model.get('name', 'Unknown')}")
            if len(models) > 5:
                print(f"     ... and {len(models) - 5} more")
        else:
            print(f"   ❌ Ollama server not responding properly")
    except Exception as e:
        print(f"   ❌ Ollama server not running: {e}")
    
    # Resource Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    
    # Memory recommendations
    if memory.total < 8 * 1024**3:  # Less than 8GB
        print(f"   ⚠️  Low RAM: {memory.total / (1024**3):.1f} GB")
        print(f"      Consider closing other applications")
    else:
        print(f"   ✅ Sufficient RAM: {memory.total / (1024**3):.1f} GB")
    
    # Disk recommendations
    if disk.free < 5 * 1024**3:  # Less than 5GB free
        print(f"   ⚠️  Low disk space: {disk.free / (1024**3):.1f} GB free")
        print(f"      Consider freeing up space")
    else:
        print(f"   ✅ Sufficient disk space: {disk.free / (1024**3):.1f} GB free")
    
    # CPU recommendations
    if psutil.cpu_count() < 4:
        print(f"   ⚠️  Limited CPU cores: {psutil.cpu_count()}")
        print(f"      Processing may be slower")
    else:
        print(f"   ✅ Adequate CPU cores: {psutil.cpu_count()}")
    
    # Missing packages
    if missing_packages:
        print(f"   ❌ Missing packages: {', '.join(missing_packages)}")
        print(f"      Run: pip install {' '.join(missing_packages)}")
    else:
        print(f"   ✅ All required packages installed")
    
    # Performance estimates
    print(f"\n⚡ PERFORMANCE ESTIMATES:")
    print(f"   Model loading: ~2-5 seconds")
    print(f"   Query processing: ~3-8 seconds")
    print(f"   Memory usage: ~2-4 GB during operation")
    print(f"   Disk usage: ~500 MB for models and index")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    issues = []
    
    if memory.total < 8 * 1024**3:
        issues.append("Low RAM")
    if disk.free < 2 * 1024**3:
        issues.append("Low disk space")
    if missing_packages:
        issues.append("Missing packages")
    if psutil.cpu_count() < 2:
        issues.append("Very limited CPU")
    
    if not issues:
        print(f"   ✅ Your system should handle the RAG chatbot well!")
    else:
        print(f"   ⚠️  Potential issues: {', '.join(issues)}")
        print(f"      Consider addressing these before running the full system")

if __name__ == "__main__":
    check_system_resources() 