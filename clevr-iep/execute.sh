python scripts/extract_features.py \
  --input_image_dir dataset_visual_bias/CLEVR_CoGenT_v1.0/images/trainA \
  --output_h5_file dataset_visual_bias/trainA_features.h5

python scripts/extract_features.py \
  --input_image_dir dataset_visual_bias/CLEVR_CoGenT_v1.0/images/valA \
  --output_h5_file dataset_visual_bias/valA_features.h5

python scripts/extract_features.py \
  --input_image_dir dataset_visual_bias/CLEVR_CoGenT_v1.0/images/testA \
  --output_h5_file dataset_visual_bias/testA_features.h5

python scripts/extract_features.py \
  --input_image_dir dataset_visual_bias/CLEVR_CoGenT_v1.0/images/valB \
  --output_h5_file dataset_visual_bias/valB_features.h5

python scripts/extract_features.py \
  --input_image_dir dataset_visual_bias/CLEVR_CoGenT_v1.0/images/testB \
  --output_h5_file dataset_visual_bias/testB_features.h5

python scripts/preprocess_questions.py \
  --input_questions_json dataset_visual_bias/CLEVR_CoGenT_v1.0/questions/CLEVR_trainA_questions.json \
  --output_h5_file dataset_visual_bias/trainA_questions.h5 \
  --output_vocab_json dataset_visual_bias/vocab.json

python scripts/preprocess_questions.py \
  --input_questions_json dataset_visual_bias/CLEVR_CoGenT_v1.0/questions/CLEVR_valA_questions.json \
  --output_h5_file dataset_visual_bias/valA_questions.h5 \
  --input_vocab_json dataset_visual_bias/vocab.json
  
python scripts/preprocess_questions.py \
  --input_questions_json dataset_visual_bias/CLEVR_CoGenT_v1.0/questions/CLEVR_testA_questions.json \
  --output_h5_file dataset_visual_bias/testA_questions.h5 \
  --input_vocab_json dataset_visual_bias/vocab.json

python scripts/preprocess_questions.py \
  --input_questions_json dataset_visual_bias/CLEVR_CoGenT_v1.0/questions/CLEVR_valB_questions.json \
  --output_h5_file dataset_visual_bias/valB_questions.h5 \
  --input_vocab_json dataset_visual_bias/vocab.json
  
python scripts/preprocess_questions.py \
  --input_questions_json dataset_visual_bias/CLEVR_CoGenT_v1.0/questions/CLEVR_testB_questions.json \
  --output_h5_file dataset_visual_bias/testB_questions.h5 \
  --input_vocab_json dataset_visual_bias/vocab.json


