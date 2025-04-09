from mne.channels import make_standard_montage
biosemi_160_montage = make_standard_montage('GSN-HydroCel-128')
biosemi_160_montage.plot()  # 默认情况下是在2D中绘制
