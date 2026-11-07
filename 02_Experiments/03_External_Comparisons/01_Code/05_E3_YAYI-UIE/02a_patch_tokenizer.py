from pathlib import Path

path=Path(__file__).resolve().parent/'model/YAYI-UIE/tokenization_YAYIUIE.py'
text=path.read_text(encoding='utf-8')
marker='        super().__init__(\n'
insert=(
    '        # Compatibility fix: initialize SentencePiece before PreTrainedTokenizer.__init__ accesses the vocabulary.\n'
    '        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)\n'
    '        self.sp_model.Load(vocab_file)\n\n'
    '        super().__init__(\n')
if 'Compatibility fix: initialize SentencePiece' in text:
    print('Tokenizer兼容补丁已存在。')
elif marker not in text:
    raise RuntimeError('未找到官方Tokenizer初始化定位点，停止修改')
else:
    path.write_text(text.replace(marker,insert,1),encoding='utf-8')
    print('Tokenizer兼容补丁已写入。')
