"""Estimate the dry fraction of a two-microphone recording from coherence and speech releases."""
import numpy as np


def dry_fraction(audio, spectrum):
    size = 160
    frames = len(audio)//size
    if frames < 32:
        return 0.
    pair = spectrum[10:110]
    cross = np.mean(pair[:,:,0]*pair[:,:,1].conj(), axis=1)
    power = np.mean(np.abs(pair)**2, axis=1)
    weights = np.sqrt(power[:,0]*power[:,1])
    coherent = np.sum(np.abs(cross)**2/np.maximum(weights,1e-12))/max(weights.sum(),1e-12)
    if coherent > .995:
        return 1.
    if coherent < .79:
        return 0.
    envelope = np.mean(audio[:frames*size].reshape(frames,size,2)**2, axis=(1,2))
    averaged = np.convolve(envelope, np.ones(3)/3, mode='valid')
    level = 10*np.log10(np.maximum(averaged, averaged.max()*1e-12+1e-30))
    start = level[:-20]
    drop = start-level[10:-10]
    falling = (start>level.max()-23) & (drop>6) & (level[20:]<start-10)
    if not np.any(falling):
        return 0.
    typical, steep = np.percentile(drop[falling], [80,95])
    if typical < 16 or steep < 18:
        return 0.
    if len(audio)%size:
        envelope = np.append(envelope, np.mean(audio[frames*size:]**2))
    level = 10*np.log10(np.maximum(envelope,1e-20)/max(envelope.max(),1e-20))
    dry = np.clip((level+34)/8, 0, 1)
    return np.interp(np.arange(len(audio)), np.arange(len(dry))*size+size/2, dry)[:,None]
