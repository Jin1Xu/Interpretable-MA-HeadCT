import gc
import json
import os
import time
import torch
import datetime
import argparse
from openai import OpenAI

from dotenv import load_dotenv 
from model_B2.base_model import create_model
from model_B2.data_loader import NiftiDataset
from torch.utils.data import DataLoader
from PIL import Image

from util import dispatch_notification, load_all_models, run_batch_screening, AGENT_ROLES

load_dotenv()
key = os.getenv("api_key")
client = OpenAI(api_key=key, base_url="https://www.dmxapi.cn/v1")

def get_args():
    parser = argparse.ArgumentParser(description="智能体参数设置")
    parser.add_argument('--model', type=str, default='local', choices=['api', 'local'],help='选择使用的模型:api或local')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B-Instruct-2507',help='选择使用的模型名称') # Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-8B, Qwen/Qwen3-VL-8B-Instruct
    parser.add_argument('--vllm_model_name', type=str, default='Qwen/Qwen3-VL-4B-Instruct',help='选择使用的模型名称')
    parser.add_argument('--top_k', type=int, default=5, help='知识库检索返回top_k条')
    parser.add_argument('--max_new_tokens', type=int, default=5000, help='大模型生成最大新token数')
    parser.add_argument('--device', type=str, default='cuda',choices=['cpu', 'cuda', 'auto'], help='推理设备: cpu, cuda, auto')
    parser.add_argument('--log_file', type=str, default='log/agent_conversations.jsonl', help='对话日志文件路径')
    parser.add_argument('--batch_size', type=int, default=5, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=1, help='数据加载的工作线程数')
    return parser.parse_args()

def agent(user_question, model_name, role_key, image_path=None):
    args = get_args()
    role_info = AGENT_ROLES[role_key]
    task = role_info['prompt']
    tk = args.top_k
    retrieved = search_knowledge_base(user_question, top_k=tk)
    
    prompt = (
        f"身份: {role_info['role']}\n"
        f"任务: \n{task}\n"
        f"问题: {user_question}\n"
    )
    prompt += f"\n\nRAG医学知识参考(不是当前患者真实情况，数据仅作参考): \n{retrieved}\n"

    if getattr(args, 'model', None) == 'local':
        try:
            use_vl = (role_key == 'observer')
            if use_vl:
                processor, model = loaded_models['v_processor'], loaded_models['llm_vl']
                content = []
                if image_path:
                    if not os.path.isfile(image_path):
                        return f"本地推理失败: 图像路径不存在 → {image_path}"
                    with Image.open(image_path) as im:
                        img_rgb = im.convert('RGB')
                        if not img_rgb.getbbox():
                            print("警告：读取到的图像是空的（全黑/全白）")
                            
                        content.append({"type": "image", "image": img_rgb})
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
        
                with torch.no_grad():
                    generated_ids = model.generate(messages, max_new_tokens=args.max_new_tokens)
                
                trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                return processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
            
            else:
                tokenizer, model = loaded_models['tokenizer'], loaded_models['llm_text']
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

                new_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
                return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
                
        except Exception as e:
            return f"本地推理失败: {str(e)}"

    else:
        try:
            api_messages = [{"role": "system", "content": role_info['prompt']}]
            


            completion = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                timeout=30 
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"API调用失败: {e}"

def resonance_collaboration_workflow(analysis_result, args):
    # analysis_result = {'disease_name': 'Composite', 'lesion_type': 'hemorrhage', 'slice_idx': 329, 'confidence': 0.45467522740364075, 'original_png': 'analysis_outputs/sample_0\\analysis_result_original.png', 'processed_png': 'analysis_outputs/sample_0\\analysis_result_processed.png', 'comparison_png': 'analysis_outputs/sample_0\\analysis_result.png'}
    image = analysis_result["processed_png"]
    print(analysis_result)
    print(image)

    if not os.path.exists(image):
        print(f"警告：文件不存在于 {image}")
    
    case_context = (
        f"""患者CT诊断结果类型：{analysis_result['disease_name']} \n"""
    )
    role_memory = {
        "observer": "尚未开始。",
        "scholar": "尚未开始。",
        "auditor": "尚未开始。"
    }
    iteration_log = []
    final_decision = "NO"
    for i in range(4):  
        
        obs_feedback = f"\n【质控/理论反馈】：{role_memory['auditor']}" if i > 0 else ""
        if i > 0:
            obs_query = (
                f"{case_context}\n你上一轮的描述：{role_memory['observer']}{obs_feedback}\n"
                "任务：请基于图像再次核实。如果图像表现确实如你所述（即使不符合典型理论），请在回复中加上‘已确认图像实测无误，理由是...’，并维持原判；若发现笔误，请修正。"
            )
        else:
            obs_query = (f"{case_context}")
        current_obs = agent(obs_query, args.model_name, 'observer', image_path=image)
        role_memory['observer'] = current_obs
        print(f'影像agent：{current_obs}\n')

        sch_feedback = ""
        if i > 0:
            sch_feedback = f"影像师意见：{current_obs}，【审核意见】：{role_memory['auditor']}"
        
        sch_query = (
            f"{case_context}\n你之前的理论基准：{role_memory['scholar']}{sch_feedback}\n"
            "任务：检索医学库。如果影像医师的观察与典型特征不符，请分析该疾病是否存在‘非典型表现’或‘分期差异’来兼容这些发现。"
        )
        current_sch = agent(sch_query, args.model_name, 'scholar')
        role_memory['scholar'] = current_sch
        print(f'理论agent：{current_sch}\n')

        audit_query = (
            f"影像医师实测：{current_obs}\n"
            f"理论专家解释：{current_sch}\n"
            "任务：两者是否存在严重的冲突？"
            "若没有回复 PASS，否则指出矛盾点回复 NO。"
        )
        current_audit = agent(audit_query, args.model_name, 'auditor')
        role_memory['auditor'] = current_audit
        print(f'质询agent：{current_audit}\n')

        if "PASS" in current_audit.upper():
            final_decision = "PASS"
            break
    
    final_report_context = (
        f"<最终影像结论>：{role_memory['observer']}\n\n"
        f"<学术理论支持>：{role_memory['scholar']}\n\n"
        f"<审核判定>：{final_decision}\n\n"
    )
    summary_report = agent(final_report_context, args.model_name, 'summarizer')
    print(iteration_log)
    return summary_report, role_memory['auditor']

if __name__ == "__main__":
    start_total = time.perf_counter()
    args = get_args()

    dataset = NiftiDataset("data")
    total_samples = len(dataset)
    predict_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    # CritiScan Agent
    model = create_model().to(args.device)
    b2_path = os.path.join("model_B2", "best_model.pth")
    if os.path.exists(b2_path):
        model.load_state_dict(torch.load(b2_path, map_location=args.device))
    model.eval()
    start_phase1 = time.perf_counter()
    abnormal_samples = run_batch_screening(model, predict_loader, args.device)
    duration_phase1 = time.perf_counter() - start_phase1
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    avg_per_sample_phase1 = duration_phase1 / total_samples if total_samples > 0 else 0.0

    print(f"初筛阶段耗时: {duration_phase1:.3f} 秒")
    print(f"处理{total_samples}个样本，平均样本耗时: {avg_per_sample_phase1:.3f}秒")

    if len(abnormal_samples) > 0:
        from huggingface_hub import login
        os.environ["HF_HOME"] = "E:\\huggingface"
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        HF_token = os.getenv("token")
        if HF_token is None:
            raise ValueError("'token'缺失")
        login(token=HF_token)

        from RAG.VKB import search_knowledge_base
        from model_B5exp.ct_analysis import run_ct_diagnosis_and_visualization

        loaded_models = load_all_models(args)

        for idx, sample_info in enumerate(abnormal_samples):
            print(f"\n分析异常样本 {idx+1}/{len(abnormal_samples)} ---")
            dispatch_notification(sample_info['prob'], idx)

            # Classify Agent
            t_start_classify = time.perf_counter()
            analysis_result = run_ct_diagnosis_and_visualization(
                nii_path=sample_info["path"],
                model_3d_path="model_B5exp/five_class/best_model.pth",
                model_2d_path="model_B5exp/explain/weight/best_model.pth",
                output_dir=f"E:\\analysis_outputs\\sample_abnormal_{idx}",
                device=args.device
            )
            total_classify_time += (time.perf_counter() - t_start_classify)
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"多分类识别结果: {analysis_result['disease_name']} (置信度: {analysis_result['confidence']:.2f})")
            
            t_start_workflow = time.perf_counter()
            summary = f"对比图路径: {analysis_result['comparison_png']}\n<检查名称>：头颅平扫CT (NCCT)\n"
            summary_text, audit_res = resonance_collaboration_workflow(analysis_result, args)
            summary += summary_text
            summary += f"""\n<诊断意见>：{analysis_result['disease_name']}\n<危急值等级>：预警"""
            total_workflow_time += (time.perf_counter() - t_start_workflow)

            print("\n" + "="*40)
            print("<最终结构化预警报告>")
            print(summary)
            print("="*40)
            
            report_path = f"warning_reports/Report_Abnormal_{idx}_{datetime.datetime.now().strftime('%H%M%S')}.txt"
            os.makedirs("warning_reports", exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"深度诊断报告已保存至: {report_path}")

        avg_classify = total_classify_time / len(abnormal_samples)
        avg_workflow = total_workflow_time / len(abnormal_samples)

        print(f"异常样本: {len(abnormal_samples)}")
        print(f"Classify Agent平均每样本耗时: {avg_classify:.3f}秒")
        print(f"Report Agent平均每样本耗时: {avg_workflow:.3f}秒")
    else:
        print("\n所有样本正常")

    print('finish')