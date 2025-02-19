import copy
import torch
from PIL import Image
from muffin.eval.muffin_vqa import init_muffin, wrap_question_with_default_conv, KeywordsStoppingCriteria
from muffin.train.train_utils import _add_speaker_and_signal, _tokenize_fn
from muffin import conversation as conversation_lib

class MuffinForSingleTurnChat:
    def __init__(self,model, img_processor, image_token_len, tokenizer) -> None:
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer

    def decode(self, image, input_ids):
        keywords = ['###']
        with torch.inference_mode():
            num_beams = 3
            input_size = input_ids.shape[-1]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_size)
            # print(f'Input: {self.tokenizer.batch_decode(input_ids)}'
            #       f'input_ids: {input_ids}')

            output = self.model.generate(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.7,
                max_new_tokens=1024,
                num_beams=num_beams,
                # do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria],
                repetition_penalty=1.1)

            response = self.tokenizer.decode(output.sequences[0][input_size:], skip_special_tokens=True)
            # print(f'raw response is {response}')
            if response.count('###'):
                response = response[: response.index('###')]
            if response.count('Assistant:'):
                response = response[response.index('Assistant:') + len('Assistant:'):]
            response = response.strip()
            return response

    def chat(self, image_path, question):
        image = Image.open(image_path).convert('RGB')
        question = wrap_question_with_default_conv(question, self.image_token_len)

        tokenized = self.tokenizer([question])
        input_ids = torch.as_tensor(tokenized['input_ids'][0])
        image = self.image_transform(image)

        return self.decode(image, input_ids)


class MuffinForMultiTurnChat(MuffinForSingleTurnChat):
    def __init__(self, model, img_processor, image_token_len, tokenizer) -> None:
        super(MuffinForMultiTurnChat, self).__init__(model, img_processor, image_token_len, tokenizer)
        self.history = []
        self.image = None

    def _update_history(self, question, out):
        self.history.append({
            'from': 'human',
            'value': question
        })
        self.history.append({
            'from': 'gpt',
            'value': out
        })

    def start_chat(self, image_path, raw_question):
        image = Image.open(image_path).convert('RGB')
        question = wrap_question_with_default_conv(raw_question, self.image_token_len)

        tokenized = self.tokenizer([question])
        input_ids = torch.as_tensor(tokenized['input_ids'][0])
        image = self.image_transform(image)

        out = self.decode(image, input_ids)
        self._update_history(raw_question, out)
        self.image = image
        return out

    def resume(self, question):
        if self.image is None or len(self.history) == 0:
            print(f'Please first start chat before resuming.')
            return ''
        conv = copy.deepcopy(self.history) + [{
            'from': 'human',
            'value': question
        }]
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conv = _add_speaker_and_signal(header, conv)
        conv = conv.strip()
        input_ids = _tokenize_fn([conv], self.tokenizer)['input_ids'][0]

        out = self.decode(self.image, input_ids)
        self._update_history(question, out)
        return out

    def clear(self):
        self.history = []
        self.image = None


# model, img_processor, image_token_len, tokenizer = init_muffin('/home/yutianyu/Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_win_SFT_combine-vqav2-train#dpo_sftwin_checked_1005-1026#dpo_sftwin_checked_1103-1106-1#1#1-beit3_large_patch16_448/checkpionts/checkpoint-20/')

if __name__ == '__main__':
    model, img_processor, image_token_len, tokenizer = init_muffin('/home/yutianyu/Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_win_SFT_combine-vqav2-train#dpo_sftwin_checked_1005-1026#dpo_sftwin_checked_1103-1106-1#1#1-beit3_large_patch16_448/checkpionts/checkpoint-20/')
    chat_model = MuffinForSingleTurnChat(model, img_processor, image_token_len, tokenizer)