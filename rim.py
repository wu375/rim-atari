import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import VideoEncoderCNN, VideoDecoderCNN

class IndependentLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            n_rims,
            bias=True,
        ):
        super(IndependentLinear, self).__init__()
        self._weights = nn.Parameter(torch.empty(n_rims, in_features, out_features), requires_grad=True)
        self._bias = nn.Parameter(torch.empty(n_rims, out_features), requires_grad=True)
        nn.init.kaiming_uniform_(self._weights, a=math.sqrt(5))
        # if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self._bias, -bound, bound)

    def forward(self, x):
        # x: (batch, n_rims, in_features)
        # assert len(x.shape) == 3, 'independent linear shape should be like (batch, n_rims, in_features)'

        x = torch.matmul(x.unsqueeze(2), self._weights).squeeze(2) + self._bias
        return x

class RIM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=6*128,
            output_size=None,
            n_rims=6,
            top_k=4,
            n_slots=1,
            n_heads=4,
            key_size=64,
            value_size=32,
            com_n_heads=4,
            com_key_size=32,
            com_value_size=32,
            num_attention_layers=1,
            attention_mlp_layers=2,
            dynamics_type='lstm',
            com_attention_mlp_layers=2,
            com_dropout=0.1,
            input_dropout=0.1,
            categorical=False,
            mask_dynamics=True, # this decides whether to run inactive blocks for communication
        ):
        super(RIM, self).__init__()

        if hidden_size % n_rims != 0:
            raise ValueError("hidden_size should be divisible by n_rims")

        self._n_rims = n_rims
        self._top_k = top_k
        self._n_heads = n_heads
        self._hidden_size = hidden_size // n_rims
        self._attention_size = self._hidden_size

        self._n_slots = n_slots
        self._n_slots_plus_one = n_slots + 1
        self._num_attention_layers = num_attention_layers
        self._attention_mlp_layers = attention_mlp_layers
        self._com_attention_mlp_layers = com_attention_mlp_layers

        self._key_size = key_size
        self._query_size = key_size

        self._value_size = value_size

        self._key_size = self._key_size * n_heads
        self._value_size = self._value_size * n_heads
        self._query_size = self._query_size * n_heads

        self._mask_dynamics = mask_dynamics

        if categorical:
            # in this case, input_size should be the number of tokens
            self._input_embed = nn.Embedding(input_size, self._attention_size * n_slots)
        else:
            self._input_embed = nn.Linear(input_size, self._attention_size * n_slots)

        self._q_linear = IndependentLinear(self._attention_size, self._query_size, n_rims=n_rims)
        self._q_layernorm = nn.LayerNorm([self._n_rims, self._query_size])
        self._k_linear = nn.Linear(self._attention_size, self._key_size)
        self._k_layernorm = nn.LayerNorm([self._n_slots_plus_one, self._key_size])
        self._v_linear = nn.Linear(self._attention_size, self._value_size)
        self._v_layernorm = nn.LayerNorm([self._n_slots_plus_one, self._value_size])

        self._input_dropout = nn.Dropout(input_dropout)

        # self._null_row = torch.zeros(1, 1, self._attention_size)
        self.register_buffer('_null_row', torch.zeros(1, 1, self._attention_size))

        self._attention_mlp = nn.ModuleList([nn.Linear(self._attention_size, self._attention_size) for _ in range(attention_mlp_layers)])
        self._attention_layernorm1 = nn.LayerNorm([n_rims, self._attention_size])
        self._attention_layernorm2 = nn.LayerNorm([n_rims, self._attention_size])

        self._dynamics_type = dynamics_type

        if dynamics_type == 'lstm':
            cell = nn.LSTMCell
        else:
            raise NotImplementedError

        self._rim_dynamics = nn.ModuleList([
            cell(input_size=self._attention_size, hidden_size=self._hidden_size) 
            for _ in range(n_rims)
        ])

        self._com_query_size = com_key_size * com_n_heads # 128
        self._com_key_size = com_key_size * com_n_heads
        self._com_value_size = com_value_size * com_n_heads

        self._com_q_linear = IndependentLinear(self._hidden_size, self._com_query_size, n_rims=n_rims)
        self._com_q_layernorm = nn.LayerNorm([n_rims, self._com_query_size])
        self._com_k_linear = IndependentLinear(self._hidden_size, self._com_key_size, n_rims=n_rims)
        self._com_k_layernorm = nn.LayerNorm([n_rims, self._com_key_size])
        self._com_v_linear = IndependentLinear(self._hidden_size, self._com_value_size, n_rims=n_rims)
        self._com_v_layernorm = nn.LayerNorm([n_rims, self._com_value_size])

        self._com_attention_mlp = [nn.Linear(self._com_value_size, self._hidden_size)]
        self._com_attention_mlp = nn.ModuleList(self._com_attention_mlp+[nn.Linear(self._hidden_size, self._hidden_size) for _ in range(com_attention_mlp_layers-1)])
        self._com_attention_layernorm1 = nn.LayerNorm([n_rims, self._com_value_size])
        self._com_attention_layernorm2 = nn.LayerNorm([n_rims, self._hidden_size])

        self._com_dropout = nn.Dropout(com_dropout)

        self._output_size = output_size
        if output_size is not None:
            self._out_linear = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size, trainable=False):
        if self._dynamics_type == 'lstm':
            # init_state = torch.stack([torch.eye(self._n_rims) for _ in range(batch_size)])
            # # pad the matrix with zeros
            # if self._hidden_size > self._n_rims:
            #     difference = self._hidden_size - self._n_rims
            #     pad = torch.zeros((batch_size, self._n_rims, difference))
            #     hx = torch.cat([init_state, pad], -1).cuda()
            #     cx = torch.cat([init_state, pad], -1).cuda()
            #     init_state = (hx, cx)
            # elif self._hidden_size < self._n_rims:
            #     hx = init_state[:, :, :self._hidden_size].cuda()
            #     cx = init_state[:, :, :self._hidden_size].cuda()
            #     init_state = (hx, cx)

            hx = torch.zeros(batch_size, self._n_rims, self._hidden_size).cuda()
            cx = torch.zeros(batch_size, self._n_rims, self._hidden_size).cuda()
            init_state = (hx, cx)
        else:
            raise NotImplementedError

        return init_state

    def _input_attention(self, elements, h, sparsify=False, return_masks=False):
        # elements: (batch, n_slots, element_size)
        # h: (batch, n_rims, hidden_size)

        k = self._k_linear(elements) # (batch, n_slots, k*heads)
        k = self._k_layernorm(k)
        v = self._v_linear(elements) # (batch, n_slots, v*heads)
        v = self._v_layernorm(v)

        # (batch, n_rims, 1, hidden_size) @ (n_rims, hidden_size, q*heads)
        q = self._q_linear(h)
        q = self._q_layernorm(q)

        n_slots = elements.shape[1]

        k = k.view(k.shape[0], n_slots, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, n_slots, k)
        v = v.view(v.shape[0], n_slots, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, n_slots, v)
        q = q.view(q.shape[0], q.shape[1], self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, n_slots, q)

        q = q * (self._key_size ** -0.5)

        qk = torch.matmul(q, k.permute(0, 1, 3, 2)) # (batch, n_heads, n_rims, q) @ (batch, n_heads, k, n_slots)
        
        # TODOa the original implementation says softmax across n_rims is better
        weights = F.softmax(qk, dim=-1) # (batch, n_heads, n_rims, n_slots)

        # TODOa in the original implementation, sparse attention is applied here:
        # for each row in each attention matrix
        # only keep top-k weights and set others to zeros
        # then normaize over sum
        #
        # This normalization should allow all gradients to flow thru values
        # So this operation only serves to sharpen the weights I think,
        # meaning that we still need to mask out gradients later in the communication layer.
        # The result is that, later at the communication layer,
        # the top-k RIMs will attend to weaker (but updated) versions of the non-top-k RIMs,
        # whereas in my implementation, they will simply attend to the old versions.
        #
        # I don't think the two implementations will differ a lot in general tho.
        null_weights = weights[:, :, :, 0] # (batch, n_heads, n_rims)
        null_weights = null_weights.mean(dim=1) # (batch, n_rims)
        kth_smallest_null_weights = torch.kthvalue(null_weights, k=self._top_k, dim=1, keepdim=True).values # (batch, 1)
        masks = null_weights <= kth_smallest_null_weights # (batch, n_rims)
        if sparsify:
            weights = weights * masks[:, None, :, None].float() # (batch, n_heads, n_rims, n_slots)
            weights = weights / weights.sum(dim=2, keepdim=True)

        attentions = torch.matmul(weights, v) # (batch, n_heads, n_rims, v)

        attentions = attentions.permute(0, 2, 1, 3).contiguous() # (batch, n_rims, n_heads, v)
        attentions = attentions.view((attentions.shape[0], attentions.shape[1], -1)) # (batch, n_rims, v*heads) = (batch, n_rims, attention_size)

        if return_masks:
            return attentions, masks.detach()

        return attentions

    def _input_attention_layer(self, elements, h):
        # h: (batch, n_rims, attention_size)
        masks = None

        if masks is None:
            attentions_new, masks = self._input_attention(elements, h, sparsify=False, return_masks=True) # (batch, n_rims, attention_size)
        else:
            attentions_new = self._input_attention(elements, h, sparsify=False, return_masks=False) # (batch, n_rims, attention_size)

        # skip connection
        # attentions_new = attentions_new + h

        attentions_new = self._attention_layernorm1(attentions_new) # (batch, n_rims, attention_size)

        new_attentions_mlp = attentions_new
        for i, _ in enumerate(self._attention_mlp):
            new_attentions_mlp = self._attention_mlp[i](new_attentions_mlp)
            new_attentions_mlp = F.relu(new_attentions_mlp)
        h = self._attention_layernorm2(attentions_new + new_attentions_mlp)

        return h, masks.float()

    def _independent_dynamics(self, attentions, h, masks, mask_dynamics=True):
        # attention: (batch, n_rims, attention_size)
        # h: tuple (hx, cx); hx, cx: (batch, n_rims, hidden_size)
        # masks: (batch, n_rims)

        hx_final = torch.zeros_like(h[0])
        cx_final = torch.zeros_like(h[1])
        # hx_final = []
        # cx_final = []
        for i, _ in enumerate(self._rim_dynamics):
            hx = h[0][:, i] # (batch, hidden_size)
            cx = h[1][:, i] # (batch, hidden_size)
            attention_k = attentions[:, i] # (batch, attention_size)
            hx_new, cx_new = self._rim_dynamics[i](attention_k, (hx, cx)) # (batch, hidden_size)

            # TODOa the original implementation applies masks after communication layer
            if mask_dynamics:
                mask = masks[:, i:i+1] # (batch, 1)
                hx_final[:, i] = hx * (1-mask) + hx_new * mask
                cx_final[:, i] = cx * (1-mask) + cx_new * mask
            else:
                hx_final[:, i] = hx_new
                cx_final[:, i] = cx_new

        h = (hx_final, cx_final)
        return h

    def _communication_attention(self, h):
        # h: (batch, n_rims, hidden_size)

        q = self._com_q_linear(h) # (batch, n_rims, q*heads)
        q = self._com_q_layernorm(q)
        k = self._com_k_linear(h) # (batch, n_rims, k*heads)
        k = self._com_k_layernorm(k)
        v = self._com_v_linear(h) # (batch, n_rims, v*heads)
        v = self._com_v_layernorm(v)

        q = q.view(q.shape[0], self._n_rims, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, n_rims, q)
        k = k.view(k.shape[0], self._n_rims, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, n_rims, k)
        v = v.view(v.shape[0], self._n_rims, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, n_rims, v)

        q = q * (self._com_key_size ** -0.5)

        qk = torch.matmul(q, k.permute(0, 1, 3, 2)) # (batch, n_heads, n_rims, q) @ (batch, n_heads, k, n_rims)
        weights = F.softmax(qk, dim=-1) # (batch, n_heads, n_rims, n_rims)

        attentions = torch.matmul(weights, v) # (batch, n_heads, n_rims, v)

        attentions = attentions.permute(0, 2, 1, 3).contiguous() # (batch, n_rims, n_heads, v)
        attentions = attentions.view((attentions.shape[0], attentions.shape[1], -1)) # (batch, n_rims, v*heads)

        return attentions

    def _communication_attention_layer(self, h, masks):
        # h: (batch, n_rims, hidden_size)
        # masks: (batch, n_rims)

        # TODOa the original implementation masks out gradient here
        masks = masks.unsqueeze(2)

        if self.training:
            h.register_hook(lambda grad: grad * masks)

        h_old = h
        # 

        for _ in range(self._num_attention_layers):
            h_new = self._communication_attention(h) # (batch, n_rims, attention_size)

            # TODOa the original implementation has a gating mechanism here
            # with linear layers

            # TODOa the original implementation does skip connection after a linear (gating) layer
            # skip connection, this requires hidden_size == attention_size
            # h_new = h_new + h

            h_new = self._com_attention_layernorm1(h_new) # (batch, n_rims, attention_size)

            new_h_mlp = h_new
            for i, _ in enumerate(self._com_attention_mlp):
                new_h_mlp = self._com_attention_mlp[i](new_h_mlp)
                new_h_mlp = F.relu(new_h_mlp)
            # h = self._com_attention_layernorm2(h_new + new_h_mlp)
            h = self._com_attention_layernorm2(h + new_h_mlp)
        
        h = h * masks + h_old * (1-masks)
        return h

    def _forward_step(self, inputs, h):
        # inputs = inputs.view(inputs.shape[0], -1) # (batch, input_size)
        # inputs = self._input_embed(inputs) # (batch, attention_size)

        if self._n_slots == 1:
            elements = inputs.unsqueeze(dim=1) # (batch, n_slots, attention_size)
        elif self._n_slots > 1:
            elements = inputs.view(inputs.shape[0], self._n_slots, inputs.shape[1]//self._n_slots)
        else:
            raise NotImplementedError

        elements = torch.cat((self._null_row.expand(elements.shape[0], *self._null_row.shape[1:]), elements), dim=1) # (batch, n_slots+1, attention_size)

        attentions, masks = self._input_attention_layer(elements, h[0]) # (batch, n_rims, attention_size), (batch, n_rims)

        h = self._independent_dynamics(attentions, h, masks=masks, mask_dynamics=self._mask_dynamics) # (batch, n_rims, hidden_size)

        hx = self._communication_attention_layer(h[0], masks) # (batch, n_rims, attention_size)
        # h = (hx, h[1])
        h = (hx, hx)

        return h, masks # (batch, n_rims, hidden_size), (batch, n_rims)

    def forward_step(self, inputs, h):
        # inputs = inputs.view(inputs.shape[0], -1) # (batch, input_size)
        inputs = self._input_embed(inputs) # (batch, attention_size)
        if self._n_slots == 1:
            elements = inputs.unsqueeze(dim=1) # (batch, n_slots, attention_size)
        else:
            raise NotImplementedError

        elements = torch.cat((self._null_row.expand(elements.shape[0], *self._null_row.shape[1:]), elements), dim=1) # (batch, n_slots+1, attention_size)

        attentions, masks = self._input_attention_layer(elements, h[0]) # (batch, n_rims, attention_size), (batch, n_rims)
        h = self._independent_dynamics(attentions, h, masks) # (batch, n_rims, hidden_size)
        # TODOa the original implementation set cx equal to hx
        hx = self._communication_attention_layer(h[0], masks) # (batch, n_rims, attention_size)
        h = (hx, h[1])

        if self._output_size is not None:
            out = self._out_linear(hx).view(hx.shape[0], -1) # (batch, n_rims*out)
            return out, h, masks
        return hx.view(hx.shape[0], -1), h, masks # (batch, n_rims, hidden_size), (batch, n_rims)

    def forward(self, inputs, h=None, return_masks=False):
        # inputs: (batch, time, input_size)
        if h is None:
            h = self.init_hidden(batch_size=inputs.shape[0]) # tuple (hx, cx); hx, cx: (batch, n_rims, hidden_size)

        inputs = self._input_embed(inputs) # (batch, time, attention_size)
        outputs = []
        all_masks = []
        for idx_step in range(inputs.shape[1]):
            h, masks = self._forward_step(inputs[:, idx_step], h) # tuple (hx, cx); (batch, n_rims, hidden_size)
            outputs.append(h[0])
            all_masks.append(masks)

        outputs = torch.stack(outputs, dim=1) # (batch, time, n_rims, hidden_size)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1) # (batch, time, n_rims*hidden_size)
        if self._output_size is not None:
            outputs = self._out_linear(outputs)

        all_masks = torch.stack(all_masks, dim=2) # (batch, n_rims, time)

        if return_masks:
            return outputs, h, all_masks
        return outputs, h

    def forward_ar(self, output, memory, encoder_fn, decoder_fn, nsteps=900):
        # output: (batch, input_size)
        # memory: (batch, mem_slots, mem_size)
        # memory = memory.detach()

        outputs = []
        for i in range(nsteps):
            decoded = decoder_fn(output.unsqueeze(dim=1)) # (batch, 1, c, w, h)
            encoded = encoder_fn(decoded)[:, 0] # (batch, hidden)
            output, memory = self._forward_step(encoded, memory)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)

        return outputs, memory

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_weights(self):
        return utils.dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)



class RIM_full_pipeline(nn.Module):
    def __init__(self, cfg):
        super(RIM_full_pipeline, self).__init__()
        self.cfg = cfg

        self.encoder = VideoEncoderCNN(
            in_channels=1,
        )
        self.decoder = VideoDecoderCNN(
            in_channels=1024,
            out_channels=1,
        )
        self.model = RIM(
            input_size=1024,
            mem_slots=4,
            head_size=64,
            n_heads=4,
            num_attention_layers=1,
            forget_bias=1.,
            input_bias=0.,
            gate_style='unit',
            attention_mlp_layers=2,
            key_size=None,
            use_adaptive_softmax=False,
            cutoffs=None,
        )
        self.loss = nn.MSELoss()

    def forward(self, obs, memory):
        obs_encoded = self.encoder(obs)
        outputs, memory = self.model(obs_encoded, memory, return_all_outputs=True)
        obs_decoded = self.decoder(outputs)
        loss = self.loss(obs[:, 1:], obs_decoded[:, :-1])
        return {
            "loss": loss,
            "obs_decoded": obs_decoded,
            "memory": memory
        }

    def forward_ar(self, obs_context, memory, nsteps=900):
        obs_encoded = self.encoder(obs_context)
        output, memory = self.model(obs_encoded, memory, return_all_outputs=False)

        outputs, memory = self.model.forward_ar(output, memory, self.encoder, self.decoder, nsteps=nsteps)
        obs_decoded = self.decoder(outputs)
        return {
            "full_recon": obs_decoded,
            "memory": memory
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_weights(self):
        return utils.dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


if __name__ == "__main__":
    model = RIM(
        input_size=32,
    )
    # encoder = VideoEncoderCNN(
    #     in_channels=1,
    # )
    # decoder = VideoDecoderCNN(
    #     in_channels=1024,
    #     out_channels=1,
    # )
    # with torch.no_grad():
    #     obs = torch.rand(8, 100, 1, 64, 64)
    #     memory = model.initial_state(batch_size=8)
    #     obs_encoded = encoder(obs)
    #     outputs, memory = model(obs_encoded, memory, return_all_outputs=True)
    #     # obs_decoded = decoder(outputs)

    #     print(outputs.shape)
    #     print(memory.shape)

    #     outputs, memory = model.forward_ar(outputs[:, -1], memory, encoder, decoder, nsteps=80)
    #     print(outputs.shape)
    #     print(memory.shape)

    #     # loss = nn.MSELoss()(obs_decoded, torch.rand(8, 100, 1, 64, 64))
    #     # print(loss)

    # h = torch.rand(8, 6, 1, 16)
    # params = nn.Parameter(torch.empty(6, 16, 32), requires_grad=True)
    # nn.init.xavier_uniform_(params)

    # prod = torch.matmul(h, params)

    # print(prod.shape)

    model.cuda()
    x = torch.rand(8, 20, 32).cuda()
    outputs, masks = model(x, return_masks=True)
    print('output shape: ', outputs.shape)
    print('masks shape: ', masks.shape)


    print('success')