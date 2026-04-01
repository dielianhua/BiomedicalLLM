# BiomedicalLLM
#1.start server
python -m vllm.entrypoints.openai.api_server \
   --model $path/$name \
  --max-model-len 40900 \
#2. predict
python predict_final.py --name medinst_"$name" --dir . --model $path/$name --key YOUR_SECRET_KEY  --base_url "http://localhost:8000/v1" >> logs/log_$name
#3. evaluation
python evaluation_final_filter.py --name medinst_"$name"  --dir . --original_data_dir ./MedINST/all_history_filter_all
