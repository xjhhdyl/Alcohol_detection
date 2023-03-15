from __future__ import division
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
# 绘图显示中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def detrendN(x,fs,m):
    N = x.size
    t = np.array([i/fs for i in range(N)])
    a = np.polyfit(t,x,m)
    xtrend = np.polyval(a,t)
    y = x - xtrend
    return y,xtrend

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i,:] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


# 短时自相关函数
def STAc(x):
    """
    计算短时相关函数
    :param x:
    :return:
    """
    para = np.zeros(x.shape)
    fn = x.shape[1]
    for i in range(fn):
        R = np.correlate(x[:, i], x[:, i], 'valid')
        para[:, i] = R
    return para

def STEn(x: object, win: object, inc: object) -> object:
    """
    计算短时能量函数
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = enframe(x, win, inc)
    s = np.multiply(X, X)
    return np.sum(s, axis=1)


def STMn(x, win, inc):
    """
    计算短时平均幅度计算函数
    :param x:
    :param win:
    :param inc:
    :return:
    """
    X = enframe(x, win, inc)
    s = np.abs(X)
    return np.mean(s, axis=1)


def STZcr(x, win, inc, delta=0):
    """
    计算短时过零率
    :param x:
    :param win:
    :param inc:
    :return:
    """
    absx = np.abs(x)
    x = np.where(absx < delta, 0, x)
    X = enframe(x, win, inc)
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    s = np.multiply(X1, X2)
    sgn = np.where(s < 0, 1, 0)
    return np.sum(sgn, axis=1)


#短时平均幅度差函数
def STAmdf(X):
    fn = X.shape[0]
    wlen = X.shape[1]
    para = np.zeros((fn,wlen))
    for i in range(fn):
        # u = np.zeros((wlen,1))
        u = X[i,:]
        for k in range(wlen):
            para[:,k] = np.sum(np.abs(u[k:wlen]-u[0:wlen-k]))
    return para.T.flatten()


def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs

#计算倒谱
def Nrceps(x):
    fftxabs = np.abs(np.fft.fft(x))
    xhat = np.real(np.fft.ifft(np.log(fftxabs)))
    return xhat

def cceps(x):
    y=np.fft.fft(x)
    return np.fft.ifft(np.log(y))

def icceps(x):
    y = np.fft.fft(x)
    return np.fft.ifft(np.exp(y))


def dct(x):
    N = len(x)
    X = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for k in range(N):
        X[k] = np.sqrt(2 / N) * np.sum(C[k] * np.multiply(x, np.cos((2 * ts + 1) * k * np.pi / 2 / N)))
    return X


def idct(X):
    N = len(X)
    x = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for n in range(N):
        x[n] = np.sqrt(2 / N) * np.sum(np.multiply(np.multiply(C[ts], X[ts]), np.cos((2 * n + 1) * np.pi * ts / 2 / N)))
    return x


def melbankm(p, NFFT, fs, fl, fh, w=None):
    """
    计算Mel滤波器组
    :param p: 滤波器个数
    :param NFFT: 一帧FFT后的数据长度
    :param fs: 采样率
    :param fl: 最低频率
    :param fh: 最高频率
    :param w: 窗(没有加窗，无效参数)
    :return:
    """
    bl = 1125 * np.log(1 + fl / 700)  # 把 Hz 变成 Mel
    bh = 1125 * np.log(1 + fh / 700)
    B = bh - bl  # Mel带宽
    y = np.linspace(0, B, p + 2)  # 将梅尔刻度等间隔
    Fb = 700 * (np.exp(y / 1125) - 1)  # 把 Mel 变成Hz
    W2 = int(NFFT / 2 + 1)
    df = fs / NFFT
    freq = [int(i * df) for i in range(W2)]  # 采样频率值
    bank = np.zeros((p, W2))
    for k in range(1, p + 1):
        f0, f1, f2 = Fb[k], Fb[k - 1], Fb[k + 1]
        n1 = np.floor(f1 / df)
        n2 = np.floor(f2 / df)
        n0 = np.floor(f0 / df)
        for i in range(1, W2):
            if n1 <= i <= n0:
                bank[k - 1, i] = (i - n1) / (n0 - n1)
            elif n0 < i <= n2:
                bank[k - 1, i] = (n2 - i) / (n2 - n0)
            elif i > n2:
                break
    return bank

def Nmfcc(x, fs, p, frameSize, inc, nfft=512, n_dct=12):
    """
    计算mfcc系数
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :return:
    """
    # 预处理-预加重
    xx = lfilter([1, -0.9375], [1], x)
    # 预处理-分幀
    xx = enframe(xx, frameSize, inc)
    # 预处理-加窗
    xx = np.multiply(xx, np.hanning(frameSize))
    # 计算FFT
    xx = np.fft.rfft(xx, nfft)
    # 计算能量谱
    xx = (np.abs(xx) ** 2)
    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nfft, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(xx, bank.T)
    # 计算DCT倒谱
    M = bank.shape[0]
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
    return mfcc


def deltacoeff(x):
    """
    计算MFCC差分系数
    :param x:
    :return:
    """
    nr, nc = x.shape
    N = 2
    diff = np.zeros((nr, nc))
    for t in range(2, nr - 2):
        for n in range(N):
            diff[t, :] += n * (x[t + n, :] - x[t - n, :])
        diff[t, :] /= 10
    return diff


def mfccf(num, s, Fs):
    """
    计算并返回信号s的mfcc参数及其一阶和二阶差分参数
    :param num:
    :param s:
    :param Fs:
    :return:
    """
    N = 512  # FFT数
    Tf = 0.02  # 窗口的时长
    n = int(Fs * Tf)  # 每个窗口的长度
    M = 24  # M为滤波器组数
    l = len(s)
    Ts = 0.01  # 帧移时长
    FrameStep = int(Fs * Ts)  # 帧移
    lifter = np.array([i for i in range(num)])
    lifter = 1 + int(num / 2) * np.sin(lifter * np.pi / num)

    if np.mean(np.abs(s)) > 0.01:
        s = s / np.max(s)
    # 计算MFCC
    mfcc = Nmfcc(s, Fs, M, n, FrameStep)

    mfcc_l = np.multiply(mfcc, lifter)
    d1 = deltacoeff(mfcc_l)
    d2 = deltacoeff(d1)
    return np.hstack((mfcc_l, d1, d2))

def lpc_coeff(s, p):
    """
    :param s: 一帧数据
    :param p: 线性预测的阶数
    :return:
    """
    n = len(s)
    # 计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))
    # 处理i=0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0
    # i=1开始，递归计算
    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    ar = np.zeros(p + 1)
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    G = np.sqrt(Ep[p - 1])
    return ar, G


def lpcff(ar, npp=None):
    """
    :param ar: 线性预测系数
    :param npp: FFT阶数
    :return:
    """
    p1 = ar.shape[0]
    if npp is None:
        npp = p1 - 1
    ff = 1 / np.fft.fft(ar, 2 * npp + 2)
    return ff[:len(ff) // 2]


def lpc_lpccm(ar, n_lpc, n_lpcc):
    lpcc = np.zeros(n_lpcc)
    lpcc[0] = ar[0]  # 计算n=1的lpcc
    for n in range(1, n_lpc):  # 计算n=2,,p的lpcc
        lpcc[n] = ar[n]
        for l in range(n - 1):
            lpcc[n] += ar[l] * lpcc[n - 1] * (n - l) / n
    for n in range(n_lpc, n_lpcc):  # 计算n>p的lpcc
        lpcc[n] = 0
        for l in range(n_lpc):
            lpcc[n] += ar[l] * lpcc[n - 1] * (n - l) / n
    return -lpcc

def rcceps(x):
    '''
    :param x: 实序列
    :return: 实数谱
    '''

    fftxabs=np.abs(np.fft.fft(x))
    if 0 in fftxabs:
        raise Exception('signal:rceps:ZeroInFFT')
    xhat=np.real(np.fft.ifft(np.log(fftxabs)))

    return xhat

def vad_revr(dst1, T1, T2):
    """
    端点检测反向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] < T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] < T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] < T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def vad_forw(dst1, T1, T2):
    """
    端点检测正向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] > T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] > T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] > T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg


def vad_TwoThr(x, wlen, inc, NIS):
    """
    使用门限法检测语音段
    :param x: 语音信号
    :param wlen: 分帧长度
    :param inc: 帧移
    :param NIS:
    :return:
    """
    maxsilence = 15
    minlen = 5
    status = 0
    y = enframe(x, wlen, inc)
    fn = y.shape[0]
    amp = STEn(x, wlen, inc)
    zcr = STZcr(x, wlen, inc, delta=0.01)
    ampth = np.mean(amp[:NIS])
    zcrth = np.mean(zcr[:NIS])
    amp2 = 2 * ampth
    amp1 = 4 * ampth
    zcr2 = 2 * zcrth
    xn = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(fn):
        if status == 0 or status == 1:
            if amp[n] > amp1:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif amp[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0

        elif status == 2:
            if amp[n] > amp2 and zcr[n] > zcr2:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        elif status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF, amp, zcr


def vad_corr(y, wnd, inc, NIS, th1, th2):
    x = enframe(y, wnd, inc)
    Ru = STAc(x.T)[0]
    Rum = Ru / np.max(Ru)
    thredth = np.max(Rum[:NIS])
    T1 = th1 * thredth
    T2 = th2 * thredth
    voiceseg, vsl, SF, NF = vad_forw(Rum, T1, T2)
    return voiceseg, vsl, SF, NF, Rum


def vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs):
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)
    X = np.abs(np.fft.fft(x, axis=1))
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    df = fs / wlen
    fx1 = int(250 // df + 1)  # 250Hz位置
    fx2 = int(3500 // df + 1)  # 500Hz位置
    km = wlen // 8
    K = 0.5
    E = np.zeros((X.shape[0], wlen // 2))
    E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
    E = np.multiply(E, E)
    Esum = np.sum(E, axis=1, keepdims=True)
    P1 = np.divide(E, Esum)
    E = np.where(P1 >= 0.9, 0, E)
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    for i in range(10):
        Hb = medfilt(Hb, 5)
    Me = np.mean(Hb)
    eth = np.mean(Hb[:NIS])
    Det = eth - Me
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    voiceseg, vsl, SF, NF = vad_revr(Hb, T1, T2)
    return voiceseg, vsl, SF, NF, Hb


def vad_pro(data, wnd, inc, NIS, thr1, thr2, mode):
    """
    使用比例法检测端点
    :param data:
    :param wnd:
    :param inc:
    :param NIS:
    :param thr1:
    :param thr2:
    :param mode:
    :return:
    """
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    if mode == 1:  # 能零比
        a = 2
        b = 1
        LEn = np.log10(1 + np.sum(np.multiply(x, x) / a, axis=1))
        EZRn = LEn / (STZcr(data, wlen, inc) + b)
        for i in range(10):
            EZRn = medfilt(EZRn, 5)
        dth = np.mean(EZRn[:NIS])
        T1 = thr1 * dth
        T2 = thr2 * dth
        Epara = EZRn
    elif mode == 2:  # 能熵比
        a = 2
        X = np.abs(np.fft.fft(x, axis=1))
        X = X[:, :wlen // 2]
        Esum = np.log10(1 + np.sum(np.multiply(X, X) / a, axis=1))
        prob = X / np.sum(X, axis=1, keepdims=True)
        Hn = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
        Ef = np.sqrt(1 + np.abs(Esum / Hn))
        for i in range(10):
            Ef = medfilt(Ef, 5)
        Me = np.max(Ef)
        eth = np.mean(Ef[NIS])
        Det = Me - eth
        T1 = thr1 * Det + eth
        T2 = thr2 * Det + eth
        Epara = Ef
    voiceseg, vsl, SF, NF = vad_forw(Epara, T1, T2)
    return voiceseg, vsl, SF, NF, Epara


def vad_LogSpec(signal, noise, NoiseCounter=0, NoiseMargin=3, Hangover=8):
    """
    对数频率距离检测语音端点
    :param signal:
    :param noise:
    :param NoiseCounter:
    :param NoiseMargin:
    :param Hangover:
    :return:
    """
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist = np.where(SpectralDist < 0, 0, SpectralDist)
    Dist = np.mean(SpectralDist)
    if Dist < NoiseMargin:
        NoiseFlag = 1
        NoiseCounter += 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0
    if NoiseCounter > Hangover:
        SpeechFlag = 0
    else:
        SpeechFlag = 1
    return NoiseFlag, SpeechFlag, NoiseCounter, Dist

def vad_revr(dst1, T1, T2):
    """
    端点检测反向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] < T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] < T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] < T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def vad_forw(dst1, T1, T2):
    """
    端点检测正向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] > T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] > T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] > T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg


def vad_TwoThr(x, wlen, inc, NIS):
    """
    使用双门限法检测语音段
    :param x: 语音信号
    :param wlen: 分帧长度
    :param inc: 帧移
    :param NIS: 无声段的帧数
    :return:
    """
    maxsilence = 15 # 设置最大无声
    minlen = 5  # 设置最小段长
    status = 0
    y = enframe(x, wlen, inc)   # 语音分帧
    fn = y.shape[0] # 取出列数
    amp = STEn(x, wlen, inc) # 计算短时能量
    zcr = STZcr(x, wlen, inc, delta=0.01)   # 计算短时过零率
    ampth = np.mean(amp[:NIS])  # 有声段短时能量的平均值
    zcrth = np.mean(zcr[:NIS])  # 有声段短时过零率平均值
    amp2 = 2 * ampth    # 门限T1
    amp1 = 4 * ampth    # 门限T2
    zcr2 = 2 * zcrth    # 门限T3
    xn = 0  # 计数器
    count = np.zeros(fn)
    silence = np.zeros(fn)
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(fn):
        if status == 0 or status == 1:
            if amp[n] > amp1:   # 若该点短时能量在T2之上，不是端点，将其标为情况2，进行下一点的检测
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1  # 语音段长加一
            elif amp[n] > amp2 or zcr[n] > zcr2:    # 若该点短时能量在T2之下，T1之上，或短时过零率在T3之上，不是端点，将其标为情况1，进行下一点的检测
                status = 1
                count[xn] += 1  # 语音段长加一
            else:   # 若该点的短时能量在所有门限之下，标为情况0，设置为无声段
                status = 0
                count[xn] = 0   # 语音段长清零
                x1[xn] = 0
                x2[xn] = 0

        elif status == 2:   # 如果是情况2
            if amp[n] > amp2 and zcr[n] > zcr2:   # 继续检测,如果同时在T1、T3之上
                count[xn] += 1  # 语音段长加一
            else:   # 否则，标为无声段
                silence[xn] += 1
                if silence[xn] < maxsilence:    # 如果无声段段长没超出限制
                    count[xn] += 1  # 语音段长加一
                elif count[xn] < minlen:    # 如果语音段长达不到最小段长限制
                    status = 0  # 回归到情况0
                    silence[xn] = 0
                    count[xn] = 0
                else:   # 语音段长超过最小段长且无声段超过限制
                    status = 3  # 标为情况3
                    x2[xn] = x1[xn] + count[xn]
        elif status == 3:   # 只有在情况3下，计数器加一
            status = 0  # 情况重新归位0
            xn += 1
            count[xn] = 0   # 设置新计数各项参数的初值
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    # 报错提示
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1   # 标出语音段
        NF[int(x1[i]):int(x2[i])] = 0   # 标出无声段
    voiceseg = findSegment(np.where(SF == 1)[0])    # 分割成语音段
    vsl = len(voiceseg.keys())  # 计算语音段的长度
    return voiceseg, vsl, SF, NF, amp, zcr

def vad_corr(y, wnd, inc, NIS, th1, th2):
    """
    用自相关函数最大值法进行端点检测
    :param y: 输入的语音数据
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param NIS: 无声段的帧数
    :param th1: 端点检测的阈值
    :param th2: 语音检测的阈值
    :return:
    """
    x = enframe(y, wnd, inc)    # 语音分帧
    Ru = STAc(x.T)[0]   # 计算短时自相关
    Rum = Ru / np.max(Ru)   # 短时自相关归一化
    thredth = np.max(Rum[:NIS])    # 找到最大值
    T1 = th1 * thredth  # 设定门限T1
    T2 = th2 * thredth  # 设定门限T2
    voiceseg, vsl, SF, NF = vad_forw(Rum, T1, T2)  # 端点检测正向比较
    return voiceseg, vsl, SF, NF, Rum


def vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs):
    """
    用谱熵法进行端点检测
    :param data: 音频数据
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param NIS: 无声段的帧数
    :param thr1: 端点检测阈值
    :param thr2: 语音检测阈值
    :param fs: 采样频率
    :return:
    """
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)  # 语音分帧
    X = np.abs(np.fft.fft(x, axis=1))   # 进行FFT变换
    if len(wnd) == 1:   # 判断是wnd是窗函数还是窗长，并以此设定窗长
        wlen = wnd
    else:
        wlen = len(wnd)
    df = fs / wlen
    fx1 = int(250 // df + 1)  # 250Hz位置
    fx2 = int(3500 // df + 1)  # 500Hz位置
    km = wlen // 8
    K = 0.5
    # 计算每帧的谱能量
    E = np.zeros((X.shape[0], wlen // 2))
    E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
    E = np.multiply(E, E)
    Esum = np.sum(E, axis=1, keepdims=True)
    P1 = np.divide(E, Esum)     # 计算每帧中每个样本点的概率密度
    E = np.where(P1 >= 0.9, 0, E)  # 将概率密度大于0.9的样本点能量设为0
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1) # 计算谱熵值
    for i in range(10):
        Hb = medfilt(Hb, 5)
    Me = np.mean(Hb)
    eth = np.mean(Hb[:NIS])
    Det = eth - Me
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    voiceseg, vsl, SF, NF = vad_revr(Hb, T1, T2)  # 端点反向检测比较
    return voiceseg, vsl, SF, NF, Hb



def vad_pro(data, wnd, inc, NIS, thr1, thr2, mode):
    """
    使用比例法检测端点
    :param data: 音频数据
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param NIS: 无声段的帧数
    :param thr1: 端点检测阈值
    :param thr2: 语音检测阈值
    :param mode: 算法模式
    :return:
    """
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)  # 语音分帧
    if len(wnd) == 1:    # 判断是wnd是窗函数还是窗长，并以此设定窗长
        wlen = wnd
    else:
        wlen = len(wnd)
    if mode == 1:  # 模式1以能零比进行端点检测
        a = 2
        b = 1
        LEn = np.log10(1 + np.sum(np.multiply(x, x) / a, axis=1))   # 式4-1-11 能量表示
        EZRn = LEn / (STZcr(data, wlen, inc) + b)   # 式4-1-12 能零比
        for i in range(10):
            EZRn = medfilt(EZRn, 5)     # 进行中值滤波
        dth = np.mean(EZRn[:NIS])
        # 门限设定
        T1 = thr1 * dth
        T2 = thr2 * dth
        Epara = EZRn
    elif mode == 2:  # 模式2以能熵比进行端点检测
        a = 2
        X = np.abs(np.fft.fft(x, axis=1))   # 对信号进行FFT运算
        X = X[:, :wlen // 2]
        Esum = np.log10(1 + np.sum(np.multiply(X, X) / a, axis=1))  # 能量表示
        prob = X / np.sum(X, axis=1, keepdims=True) # 概率密度
        Hn = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
        Ef = np.sqrt(1 + np.abs(Esum / Hn))  # 式4-1-13 能熵比
        for i in range(10):
            Ef = medfilt(Ef, 5)     # 进行中值滤波
        Me = np.max(Ef)  # 找到能熵比的最大值
        # 门限设定
        eth = np.mean(Ef[NIS])
        Det = Me - eth
        T1 = thr1 * Det + eth
        T2 = thr2 * Det + eth
        Epara = Ef
    voiceseg, vsl, SF, NF = vad_forw(Epara, T1, T2)
    return voiceseg, vsl, SF, NF, Epara

def vad_LogSpec(signal, noise, NoiseCounter=0, NoiseMargin=2, Hangover=8):
    """
    对数频率距离检测语音端点
    :param signal:
    :param noise:
    :param NoiseCounter:
    :param NoiseMargin:
    :param Hangover:
    :return:
    """
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist = np.maximum(SpectralDist,0)

    Dist = np.mean(SpectralDist)
    if Dist < NoiseMargin:
        NoiseFlag = 1
        NoiseCounter += 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0
    if NoiseCounter > Hangover:
        SpeechFlag = 0
    else:
        SpeechFlag = 1
    return NoiseFlag, SpeechFlag, NoiseCounter, Dist

def pitch_vad(x, wnd, inc, T1, miniL=10):
    """
    使用能熵比检测基音，实际上就是语音分段
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 端点检测阈值
    :param miniL: 语音段最小帧数
    :return:
    """
    y = enframe(x, wnd, inc)    # 语音分帧
    fn = y.shape[0]
    if isinstance(wnd, int):    # 判断是wnd是窗函数还是窗长，并以此设定窗长
        wlen = wnd
    else:
        wlen = len(wnd)

    Sp = np.abs(np.fft.fft(y, axis=1))  # FFT变换
    Sp = Sp[:, :wlen // 2 + 1]  # 取正频率部分
    Esum = np.sum(np.multiply(Sp, Sp), axis=1)  # 能量总和
    prob = Sp / np.sum(Sp, axis=1, keepdims=True)   # 概率密度
    H = -np.sum(np.multiply(prob, np.log10(prob + 1e-16)), axis=1)  # 短时熵谱
    H = np.where(H < 0.1, np.max(H), H)
    Ef = np.sqrt(1 + np.abs(Esum / H))  # 能熵比
    Ef = Ef / np.max(Ef)    # 归一化

    zseg = findSegment(np.where(Ef > T1)[0])    # 找到能熵比大于门限的语音帧
    zsl = len(zseg.keys()) # 计算帧长
    SF = np.zeros(fn)
    # 标记有声段、噪声段
    for k in range(zsl):
        if zseg[k]['duration'] < miniL:
            zseg.pop(k)
        else:
            SF[zseg[k]['start']:zseg[k]['end']] = 1
    return zseg, len(zseg.keys()), SF, Ef

def pitch_Ceps(x, wnd, inc, T1, fs, miniL=10):
    """
    倒谱法基音周期检测函数
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 门限
    :param fs: 采样率
    :param miniL: 语音段的最小帧数
    :return:
    """
    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)  # 语音分段
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    y1 = y[np.where(SF == 1)[0], :]  # 找到语音段
    y1 = np.multiply(y1, np.hamming(wlen))  # 语音段加汉明窗
    xx = np.fft.fft(y1, axis=1)     # 进行FFT变换
    b = np.fft.ifft(2 * np.log(np.abs(xx) + 1e-10))   # 计算倒谱
    Lc = np.argmax(b[:, lmin:lmax], axis=1) + lmin - 1  # 在设定的倒频率间找到倒谱的最大值对应的样本点
    period[np.where(SF == 1)[0]] = Lc   # 标记基音周期
    return voiceseg, vsl, SF, Ef, period


def pitch_Corr(x, wnd, inc, T1, fs, miniL=10):
    """
    自相关法基音周期检测函数
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 门限
    :param fs: 采样率
    :param miniL: 语音段的最小帧数
    :return:
    """
    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)  # 语音分段
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    for i in range(vsl):    # 在所有语音段中
        ixb = voiceseg[i]['start']
        ixd = voiceseg[i]['duration']
        for k in range(ixd):
            ru = np.correlate(y[k + ixb, :], y[k + ixb, :], 'full')  # 计算短时自相关
            ru = ru[wlen:]
            tloc = np.argmax(ru[lmin:lmax])     # 找到范围内短时自相关最大值的标号
            period[k + ixb] = lmin + tloc  # 将其作为基音周期的估值
    return voiceseg, vsl, SF, Ef, period


def pitch_Lpc(x, wnd, inc, T1, fs, p, miniL=10):
    """
    线性预测法基音周期检测函数
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 门限
    :param fs: 采样率
    :param p: 线性预测阶数
    :param miniL: 语音段的最小帧数
    :return:
    """

    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)   # 语音分段
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    for k in range(y.shape[0]):
        if SF[k] == 1:
            u = np.multiply(y[k, :], np.hamming(wlen))  # 加汉明窗
            ar, _ = lpc_coeff(u, p)  # 计算p阶线性预测的系数
            ar[0] = 0
            z = lfilter(-ar, [1], u)    # 构造线性预测滤波器
            E = u - z   # 误差信号
            # 计算误差的倒谱
            xx = np.fft.fft(E)
            b = np.fft.ifft(2 * np.log(np.abs(xx) + 1e-20))
            lc = np.argmax(b[lmin:lmax])    # 找到倒谱最大值的标号
            period[k] = lc + lmin   # 将其作为基音周期的估值
    return voiceseg, vsl, SF, Ef, period

def local_maxium(x):
    """
    求序列的极大值
    :param x: 输入序列
    :return:
    """
    d = np.diff(x)  # 计算离散差分
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d - 1):
        if d[i] > 0 and d[i + 1] <= 0:   # 找出序列中比左右两边都大的数值，即极大值
            maxium.append(x[i + 1])
            loc.append(i + 1)
    return maxium, loc


def Formant_Cepst(u, cepstL):
    """
    倒谱法共振峰估计函数
    :param u: 一帧语音输入数据
    :param cepstL: 倒频率上的窗函数宽度
    :return:
    """
    wlen2 = len(u) // 2
    U = np.log(np.abs(np.fft.fft(u)[:wlen2]))   # 对信号进行傅里叶变换
    Cepst = np.fft.ifft(U)  # 计算其倒谱
    cepst = np.zeros(wlen2, dtype=np.complex)   # 给倒谱信号加窗
    cepst[:cepstL] = Cepst[:cepstL]
    cepst[-cepstL + 1:] = Cepst[-cepstL + 1:]
    spec = np.real(np.fft.fft(cepst))  # 计算包络谱
    val, loc = local_maxium(spec)  # 输出共振峰的幅值和位置
    return val, loc, spec


def Formant_Interpolation(u, p, fs):
    """
    插值法估计共振峰函数
    :param u: 一帧语音输入数据
    :param p: LPC阶数
    :param fs: 采样率
    :return:
    """
    ar, _ = lpc_coeff(u, p)    # 计算p阶线性预测器的系数
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)  # 计算其离散傅里叶变换
    df = fs / 512   # 谱图的频率间隔
    val, loc = local_maxium(U)  # loc存放共振峰的位置
    ll = len(loc)
    pp = np.zeros(ll)   # 存储共振峰幅值
    F = np.zeros(ll)    # 存储共振峰频率
    Bw = np.zeros(ll)   # 存储共振峰带宽
    for k in range(ll):
        m = loc[k]  # 定位第k个共振峰
        m1, m2 = m - 1, m + 1   # 找到其左右两边的相邻点
        p = val[k]  # 计算此共振峰幅值
        p1, p2 = U[m1], U[m2]   # 给出左右相邻点的幅值
        # 计算拟合函数的系数
        aa = (p1 + p2) / 2 - p
        bb = (p2 - p1) / 2
        cc = p
        dm = -bb / 2 / aa   # 极大值
        pp[k] = -bb * bb / 4 / aa + cc  # 中心频率对应的功率谱
        m_new = m + dm  # 共振峰的中心频率
        bf = -np.sqrt(bb * bb - 4 * aa * (cc - pp[k] / 2)) / aa     # 带宽的计算
        F[k] = (m_new - 1) * df
        Bw[k] = bf * df
    return F, Bw, pp, U, loc


def Formant_Root(u, p, fs, n_frmnt):
    """
    LPC求根法的共振峰估计函数
    :param u: 一帧语音输入数据
    :param p: LPC阶数
    :param fs: 采样率
    :param n_frmnt: 待求的共振峰个数
    :return:
    """
    ar, _ = lpc_coeff(u, p)     # 计算p阶线性预测器的系数
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)  # 计算其离散傅里叶变换
    const = fs / (2 * np.pi)    # 转换成角频率
    rts = np.roots(ar)  # 线性预测多项式的根
    yf = []     # 存储共振峰频率
    Bw = []     # 存储共振峰带宽
    for i in range(len(ar) - 1):
        re = np.real(rts[i])    # 记录根的实部
        im = np.imag(rts[i])    # 记录根的虚部
        fromn = const * np.arctan2(im, re)  # 计算Fi
        bw = -2 * const * np.log(np.abs(rts[i]))    # 计算Bi
        if fromn > 150 and bw < 700 and fromn < fs / 2: # 满足条件的值看作共振峰的频率和带宽
            yf.append(fromn)
            Bw.append(bw)
    return yf[:min(len(yf), n_frmnt)], Bw[:min(len(Bw), n_frmnt)], U
