import os
import json
import torch
import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

AGENT_ROLES = {
    "CritiScan Agent": { 
        "name": "初步诊断智能体",
        "role": "危急值初步分类专家",
        "prompt": """负责实时危急值筛选，如果危急值高则预警"""
    },
    "Classify Agent": {
        "name": "多分类智能体",
        "role": "多分类病症识别专家",
        "prompt": "根据影像特征，识别可能的多种病症及其概率。"
    },
    "observer": {
        "name": "影像医师(视觉)",
        "role": "AI影像视觉解析专家",
        "prompt": """【核心任务】
                    1. 解剖定侧：严格采用放射学标准体位原则，考虑影像是左右反的但是上下不反，病灶区域定位要注意。
                    2. 定量描述：尺寸为基于影像比例的保守估计。必须提供病灶的最长径和最大厚度（单位：mm）。若存在中线偏移，须明确偏移方向及具体距离（mm）。若存在偏移，其幅度极小，处于“可忽略/不构成中线偏移”的范围内。
                    3. 无需输出其他无关内容
                    
                    【影像注释要点】
                    - 解剖部位：如颞叶、顶叶、枕叶、基底节区、丘脑、额叶、颅板下、脑干、岛叶、等，并标明患者的左/右而不是图像的左右；
                    - 形态：圆形、椭圆形、不规则形、团块状等；
                    - 边界：清晰 / 模糊；
                    - 密度：高/低/混合密度，是否均匀，必要时估算CT值（HU）；
                    - 大小：估计最长径、最大厚度、短径，单位 mm；
                    - 中线结构：是否偏移？方向？偏移距离（mm）？
                    - 脑室系统：是否受压、变形？有无血肿破入脑室？
                    - 脑疝：有无钩回疝、小脑幕切迹疝等征象？
                    - 占位效应：邻近脑沟变浅、脑室受压、水肿带等；
                    - 脑积水：有无脑室扩张等表现；
                    - 颅骨：有无骨折、骨质破坏？


                    【输出格式参考（仅作结构示范，不得照搬内容）】
                    示例（基于真实报告，仅用于说明应包含的要素）：
                    解剖部位：：患者左侧（图像右侧侧）基底节区可见高密度病灶。  
                    形态：不规则形/类椭圆形团块状病灶  
                    边界：边界相对清晰  
                    密度：高密度病灶，密度欠均匀，CT值估测约50–70 HU  
                    大小：最长径约40–45 mm，最大厚度约25–30 mm  
                    中线结构：轻度向右侧偏移，约3–5 mm  
                    脑室系统：左侧侧脑室体部受压变形，未见血肿破入脑室  
                    脑疝：未见明确脑疝征象  
                    占位效应：明显，表现为侧脑室受压、邻近脑沟变浅、周围低密度水肿带  
                    脑积水：无  
                    颅骨：未见骨折
                    并根据知识库指出患者危急判定的理由

                    【严禁项】
                    - 禁止幻觉：仅描述图像中可明确识别的结构与异常；未见即写“未见”或“无”，严禁推测或编造。
                    - 禁止复制示例内容：上述示例仅为格式与要素指引，实际输出必须完全基于当前图像所见。
                    - 如果图像质量极差无法辨认，请说明原因
                    """
    },
    "scholar": {
        "name": "理论专家(知识)",
        "role": "医学影像理论专家",
        "prompt": """
                【核心任务】
                针对指定的诊断结果，基于医学知识库输出“理论特征基准”。

                【输出要求：精准范围与关联】
                1. 范围判定：在[密度]、[大小]、[偏移距离]等定量项上，必须给出医学统计上的常见值范围（如：50-80 HU，>5mm等）。
                2. 逻辑关联：必须指出该病症的“共生征象”。例如：“高密度团块状”通常关联“占位效应”和“周围水肿”。
                3. 观点精简：严禁废话和自由发散，直接输出结论。

                【影像注释要点输出规范】
                - 解剖部位：[指出该病种的好发部位，如：基底节、丘脑]
                - 形态：[如：典型呈透镜状/梭形/新月形]
                - 边界：[如：清晰且锐利]
                - 密度：[给出典型CT值范围，如：60-90 HU]
                - 大小：[给出病情严重程度的理论阈值，如：>30ml或最长径>3cm提示手术指征]
                - 中线偏移：[关联逻辑：若病灶厚度>10mm，常伴随>5mm偏移]
                - 脑室/脑疝/积水：[关联逻辑：描述该病种导致这些并发症的理论概率或解剖学基础]
                - 颅骨关联：[如：硬膜外血肿须强制核查是否存在线性骨折]
                
                并推测该病症可能的患病理由
                """
    },
    "auditor": {
        "name": "质询员(逻辑)",
        "role": "逻辑冲突检测专家",
        "prompt": """【核心任务】
                对比[影像医师]的实测描述与[理论专家]的典型基准，识别出两者之间存在的“统计学偏离”或“逻辑漏项”。此外也要识别[影像医师]发言本身自相矛盾的部分

                【审查原则】
                1. 影像优先原则：患者个体差异巨大，允许实测值偏离理论。仅当差距极显著（如高密度血肿测出低密度水肿值）时才提出质疑。
                2. 委婉质询：对于非原则性矛盾，采用“请核实”或“是否伴随”的口吻，由影像医师做最终判定。
                3. 左右定侧是红线：若解剖方位存在空间逻辑错误，必须严格标注。
                4. 逻辑自洽复核：重点审查是否存在“定侧错误”（如描述左侧病灶，后面却写右侧脑室受压）或“常识性漏项”。
                5. 质疑而非强制：对于不符合典型理论的表现，应以“质询”形式提示影像医师复核，而不是直接否定。

                PASS 判定标准】
                - 满足以下任一条件即可回复 "PASS"：
                1. 影像描述与理论基准基本吻合。
                2. 影像描述虽与典型理论不符，但影像医师在回复中明确表示“已复核图像，确认实测无误”。
                
                【输出格式】
                - 没有严重冲突：回复 "PASS"，并分点简要阐述冲突
                - 存在严重冲突：回复 "NO"，并分点简要阐述冲突
                如<中线偏移> ：
                """
    },
    "reporter": {
        "name": "结构化报告智能体",
        "role": "放射科首席医师",
        "prompt": """你负责根据校验后的共识生成正式的<结构化报告>。
        若接收到的<审核判定>：为"NO"则优先以影像描述为主。
        必须严格遵守以下格式，按照【影像注释要点】
        格式如下：""<影像表现>：
        解剖部位：基底节区、额部、颅板下方、脑干等；左、右等；
        形态：圆形、椭圆形、不规则形、团块状等；
        边界：边界清除、边界模糊
        密度：密度均匀、密度不均匀、高密度、低密度、混合密度、CT值（HU）；
        大小：最长径（mm）、最大厚度（mm）、短径（mm）；
        是否伴随中线偏移及偏移距离（mm）；
        是否伴脑室出血或血肿破入脑室系统；
        是否伴脑疝；
        是否伴占位效应（如病灶邻近脑沟、脑室受压）；
        是否伴脑积水；
        是否伴颅骨骨折。

        并综合知识库、影像专家和理论专家的发言指出患者危急判定的理由
        ""不要输出其他多余内容
        """
    }
}

def dispatch_notification(prob, sample_idx):
    """模拟流程图中的三方并行通知"""
    print("\n" + ">>>" * 10)
    print(f"""[多方预警系统已触发 - 样本 {sample_idx+1}]危急概率为: {100*prob:.4f}%
            <致技师>：危急值预警！可能存在危急值，请密切关注患者病情变化，确保检查安全。
            <致临床医生>：危急值预警！可能存在危急值，请密切关注患者病情变化，等待放射医师复核及正式报告。
            <致放射医生>：危急值预警！疑似危急值报告已置顶，请优先复核。
            请放射科医生复核确认：是否属于危急值并启动报告撰写""")
    print(">>>" * 10 + "\n")

def log_conversation(log_file, idx, user_question, ct_prob, answer):
    """将单次问答记录写入 JSONL 格式日志文件"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sample_index": idx,
        "user_question": user_question,
        "ct_prob": ct_prob,
        "answer": answer
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_all_models(args):
    models = {}
    print("正在预加载LLM...")

    if args.model == 'local':
        print(f"加载文本模型: {args.model_name}")
        models['tokenizer'] = AutoTokenizer.from_pretrained(args.model_name)
        models['llm_text'] = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype="auto", device_map=args.device
        )

        print(f"加载视觉模型: {args.vllm_model_name}")
        models['v_processor'] = AutoProcessor.from_pretrained(args.vllm_model_name)
        models['llm_vl'] = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vllm_model_name, torch_dtype="auto", device_map=args.device
        )
    
    return models

def get_ct_prob(model, image_tensor, device='cuda'):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor.unsqueeze(0))  # (1, num_classes)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        print({"normal": float(prob[0]), "abnormal": float(prob[1])})
    return {"normal": float(prob[0]), "abnormal": float(prob[1])}

def run_batch_screening(model, data_loader, device, threshold=0.5):
    abnormal_queue = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="初筛中", unit="batch"):
            images = batch['image'].to(device)
            paths = batch['path']
            labels = batch['label']
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            abnormal_probs = probs[:, 1].cpu().numpy()
            
            for i in range(len(abnormal_probs)):
                prob = float(abnormal_probs[i])
                current_path = paths[i]
                
                if prob >= threshold:
                    print(f"[发现异常] 样本: {os.path.basename(current_path)} | 概率: {prob:.4f}")
                    abnormal_queue.append({
                        'path': current_path,
                        'prob': prob,
                        'label': labels[i].item()
                    })
                    
    print(f"总样本数: {len(data_loader.dataset)}, 发现疑似异常: {len(abnormal_queue)}")
    return abnormal_queue

