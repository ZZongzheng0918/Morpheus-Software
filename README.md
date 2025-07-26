<div align="center">
<h2>Morpheus: A Neural-driven Animatronic Face with<br>
  Hybrid Actuation and Diverse Emotion Control</h2>

  **RSS 2025**
<h2>Software</h2>


**Zongzheng Zhang**<sup>1,2*</sup> · **Jiawen Yang**<sup>1*</sup> · [**Ziqiao Peng**](https://ziqiaopeng.github.io/)<sup>1</sup> ·<br>
**Meng Yang**<sup>4</sup> · [**Jianzhu Ma**](https://majianzhu.com/)<sup>1</sup>, **Lin Cheng**<sup>5</sup> · [**Huazhe Xu**](http://hxu.rocks/)<sup>3</sup> . [**Hang Zhao**](https://hangzhaomit.github.io/)<sup>3</sup> and [**Hao Zhao**](https://sites.google.com/view/fromandto/)<sup>1,2</sup><br>

<sup>1</sup> Institute for AI Industry Research (AIR), Tsinghua University, <sup>2</sup> Beijing Academy of Artificial Intelligence (BAAI),<br>
<sup>3</sup> Institute for Interdisciplinary Information Sciences(IIIS), Tsinghua University, <br>
<sup>4</sup> MGI Tech, Shenzhen, China, <sup>5</sup> Beihang University<br>
<sub>(* indicates equal contribution)</sub><br>
[**RSS official**](https://roboticsconference.org/program/papers/80/) | [**Project Page**](https://jiawenyang-ch.github.io/Morpheus-Hardware-Design/)

</div>

## Environment

- Linux
- Python 3.10
- Pytorch 2.3.1
- CUDA 12.1
- Blender 3.4.1
- ffmpeg 4.4.1

Clone the repo:
  ```bash
  git clone https://github.com/ZZongzheng0918/Morpheus-Software.git
  cd Morpheus-Software
  ```  
Create conda environment:
```bash
conda create -n morpheus python=3.10
conda activate morpheus
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## **Demo**
Download Blender and put it in this directory.
```bash
wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender3.4/blender-3.4.1-linux-x64.tar.xz
tar -xf blender-3.4.1-linux-x64.tar.xz
mv blender-3.4.1-linux-x64 blender && rm blender-3.4.1-linux-x64.tar.xz
```
Download the pretrained models from [model.pth](https://huggingface.co/ZiqiaoPeng/Morpheus). Put the pretrained models under `pretrain_model` folder. 
Put the audio under `aduio` folder and run
```bash
python demo.py --wav_path "./audio/disgust.wav"
```
The generated animation will be saved in `result` folder.



## **License and Acknowledgements**
This source code is licensed under the MIT liscence found in the LICENSE file in the root directory of this repository.



## **Citation**
If you find this project useful, feel free to cite our work!
<div style="display:flex;">
<div>

```bibtex
@article{zhang2025morpheus,
  title={Morpheus: A Neural-driven Animatronic Face with Hybrid Actuation and Diverse Emotion Control},
  author={Zhang, Zongzheng and Yang, Jiawen and Peng, Ziqiao and Yang, Meng and Ma, Jianzhu and Cheng, Lin and Xu, Huazhe and Zhao, Hang and Zhao, Hao},
  journal={arXiv preprint arXiv:2507.16645},
  year={2025}
}
```

## **Acknowledgement**
Here are some great resources we benefit:
- [EmoTalk](https://github.com/psyai-net/EmoTalk_release) for codebase
- [Faceformer](https://github.com/EvelynFan/FaceFormer) for training pipeline
- [EVP](https://github.com/jixinya/EVP) for training dataloader
- [Speech-driven-expressions](https://github.com/YoungSeng/Speech-driven-expressions) for rendering
- [Wav2Vec2 Content](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) and [Wav2Vec2 Emotion](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition) for audio encoder
- [Head Template](http://filmicworlds.com/blog/solving-face-scans-for-arkit/) for visualization.

Thanks to John Hable for sharing his head template under the CC0 license, which is very helpful for us to visualize the results.
