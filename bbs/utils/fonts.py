import matplotlib.pyplot as plt
import platform

def setup_matplotlib_fonts():
    if platform.system() == 'Linux':
        plt.rcParams['font.family'] = 'NanumGothic'
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
 
    plt.rcParams['axes.unicode_minus'] = False
