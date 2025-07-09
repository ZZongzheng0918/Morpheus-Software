import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from fairseq.models.speech_to_text import Conv1dSubsampler
from tqdm import tqdm
from transformers import Wav2Vec2Processor,Wav2Vec2FeatureExtractor
from numpy.linalg import norm

from dtw import dtw
from wav2vec import Wav2Vec2Model,Wav2Vec2ForSpeechClassification
import argparse

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S).to(device)
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=30, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class A2BModel(nn.Module):
    def __init__(self,args):
        super(A2BModel, self).__init__()
        self.feature_dim = args.feature_dim
        self.bs_dim = args.bs_dim
        self.device = args.device
        self.batch_size = args.batch_size
        self.audio_encoder_cont = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.audio_encoder_cont.feature_extractor._freeze_parameters()
        self.audio_encoder_emo = Wav2Vec2ForSpeechClassification.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
        self.audio_encoder_emo.wav2vec2.feature_extractor._freeze_parameters()
        self.max_seq_len = args.max_seq_len
        self.emo_guide = args.emo_guide
        self.audio_feature_map_cont = nn.Linear(1024, 512)
        self.audio_feature_map_emo = nn.Linear(1024, 832)
        self.audio_feature_map_emo2 = nn.Linear(832, 256)
        self.relu = nn.ReLU()

        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period, max_seq_len = args.max_seq_len)
        self.biased_mask1 = init_biased_mask(n_head = 4, max_seq_len = args.max_seq_len, period=args.period)
        self.one_hot_level = np.eye(2)
        self.obj_vector_level = nn.Linear(2, 32)
        self.one_hot_person = np.eye(24)
        self.obj_vector_person = nn.Linear(24, 32)

        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.bs_map_r = nn.Linear(self.feature_dim, self.bs_dim)
        nn.init.constant_(self.bs_map_r.weight, 0)
        nn.init.constant_(self.bs_map_r.bias, 0)

    def forward(self,data):
        frame_num11 = data["target11"].shape[1]
        frame_num12 = data["target12"].shape[1]
        frame_num21 = data["target21"].shape[1]
        inputs11 = self.processor(torch.squeeze(data["input11"]), sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(self.device)
        inputs12 = self.processor(torch.squeeze(data["input12"]), sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(self.device)

        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num12).last_hidden_state
        if frame_num12 != frame_num11:
            hidden_states_cont11 = self.audio_encoder_cont(inputs11, frame_num=frame_num11).last_hidden_state.squeeze(0)
            hidden_states_cont1 = hidden_states_cont1.squeeze(0)
            dist, cost, acc_cost, path = dtw(hidden_states_cont1, hidden_states_cont11, dist=lambda x, y: torch.norm(x - y, p=1))
            hidden_states_cont1_n = hidden_states_cont11
            a = path[0]
            b = path[1]
            for l in range(1, len(path[0])):
                hidden_states_cont1_n[b[l]] = hidden_states_cont1[a[l]]
            hidden_states_cont1 = hidden_states_cont1_n.unsqueeze(0)
        hidden_states_cont12 = self.audio_encoder_cont(inputs12, frame_num=frame_num12).last_hidden_state

        inputs11 = self.feature_extractor(torch.squeeze(data["input11"]), sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(self.device)
        inputs21 = self.feature_extractor(torch.squeeze(data["input21"]), sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(self.device)
        inputs12 = self.feature_extractor(torch.squeeze(data["input12"]), sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(self.device)

        output_emo11 = self.audio_encoder_emo(inputs11, frame_num=frame_num11)
        output_emo1 = self.audio_encoder_emo(inputs21, frame_num=frame_num21)
        output_emo2 = self.audio_encoder_emo(inputs12, frame_num=frame_num12)

        hidden_states_emo1 = output_emo1.hidden_states
        if frame_num21 != frame_num11:
            hidden_states_emo11 = output_emo11.hidden_states.squeeze(0)
            hidden_states_emo1 = hidden_states_emo1.squeeze(0)
            dist, cost, acc_cost, path = dtw(hidden_states_emo1, hidden_states_emo11, dist=lambda x, y: torch.norm(x - y, p=1))
            hidden_states_emo1_n = hidden_states_emo11
            a = path[0]
            b = path[1]
            for l in range(1, len(path[0])):
                hidden_states_emo1_n[b[l]] = hidden_states_emo1[a[l]]
            hidden_states_emo1 = hidden_states_emo1_n.unsqueeze(0)
        hidden_states_emo2 = output_emo2.hidden_states

        label1 = output_emo1.logits
        onehot_level = self.one_hot_level[data["level"]]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[data["person"]]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        if data["target11"].shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1,0,2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1,frame_num11,1)
        obj_embedding_level12 = obj_embedding_level.repeat(1,frame_num12,1)
        obj_embedding_person11 = obj_embedding_person.repeat(1,frame_num11,1)
        obj_embedding_person12 = obj_embedding_person.repeat(1,frame_num12,1)
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1) #hidden_states[32,540,64]
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1) #hidden_states[32,540,64]
        hidden_states_emo11_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo11_832)) #hidden_states[32,540,64]


        hidden_states11 = torch.cat([hidden_states_cont1, hidden_states_emo11_256,obj_embedding_level11,obj_embedding_person11], dim=2)
        hidden_states_cont12 = self.audio_feature_map_cont(hidden_states_cont12) #hidden_states[32,540,64]
        hidden_states_emo12_832 = self.audio_feature_map_emo(hidden_states_emo2) #hidden_states[32,540,64]
        hidden_states_emo12_256 = self.relu(self.audio_feature_map_emo2(hidden_states_emo12_832)) #hidden_states[32,540,64]


        hidden_states12 = torch.cat([hidden_states_cont12, hidden_states_emo12_256,obj_embedding_level12,obj_embedding_person12], dim=2)
        if data["target11"].shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1], :hidden_states11.shape[1]].clone().detach().to(device=self.device)
            tgt_mask22 = self.biased_mask1[:, :hidden_states12.shape[1], :hidden_states12.shape[1]].clone().detach().to(device=self.device)
        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])


        memory_mask12 = enc_dec_mask(self.device, hidden_states12.shape[1], hidden_states12.shape[1])
        if self.emo_guide:
            bs_out11 = self.transformer_decoder(hidden_states11, hidden_states_emo11_832, tgt_mask=tgt_mask11, memory_mask=memory_mask11)
            bs_out12 = self.transformer_decoder(hidden_states12, hidden_states_emo12_832, tgt_mask=tgt_mask22, memory_mask=memory_mask12)
        else:
            bs_out11 = self.transformer_decoder(hidden_states11, hidden_states11, tgt_mask=tgt_mask11, memory_mask=memory_mask11)
            bs_out12 = self.transformer_decoder(hidden_states12, hidden_states12, tgt_mask=tgt_mask22, memory_mask=memory_mask12)


        bs_output11 = self.bs_map_r(bs_out11) 
        bs_output12 = self.bs_map_r(bs_out12) 


        return bs_output11,bs_output12,label1


    def predict(self, audio,level,person):
        frame_num11 = math.ceil(audio.shape[1]/16000*30) 
        inputs12 = self.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=frame_num11).last_hidden_state

        inputs12 = self.feature_extractor(torch.squeeze(audio), sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.audio_encoder_emo(inputs12, frame_num=frame_num11)
        hidden_states_emo1 = output_emo1.hidden_states

        onehot_level = self.one_hot_level[level]
        onehot_level = torch.from_numpy(onehot_level).to(self.device).float()
        onehot_person = self.one_hot_person[person]
        onehot_person = torch.from_numpy(onehot_person).to(self.device).float()
        if audio.shape[0] == 1:
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0)
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0)
        else:
            obj_embedding_level = self.obj_vector_level(onehot_level).unsqueeze(0).permute(1,0,2)
            obj_embedding_person = self.obj_vector_person(onehot_person).unsqueeze(0).permute(1, 0, 2)

        obj_embedding_level11 = obj_embedding_level.repeat(1, frame_num11, 1)
        obj_embedding_person11 = obj_embedding_person.repeat(1, frame_num11, 1)
        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)  
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.audio_feature_map_emo2(hidden_states_emo11_832))  

        hidden_states11 = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256, obj_embedding_level11, obj_embedding_person11], dim=2)
        if audio.shape[0] == 1:
            tgt_mask11 = self.biased_mask1[:, :hidden_states11.shape[1],
                         :hidden_states11.shape[1]].clone().detach().to(device=self.device)
        memory_mask11 = enc_dec_mask(self.device, hidden_states11.shape[1], hidden_states11.shape[1])
        if self.emo_guide:
            bs_out11 = self.transformer_decoder(hidden_states11,hidden_states_emo11_832, tgt_mask=tgt_mask11, memory_mask=memory_mask11)
        else:
            bs_out11 = self.transformer_decoder(hidden_states11, hidden_states11, tgt_mask=tgt_mask11, memory_mask=memory_mask11)
        bs_output11 = self.bs_map_r(bs_out11) 

        return bs_output11

