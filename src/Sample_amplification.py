# ./sft_samples/data/文件夹下存放了12个文件夹（目前仅有6个），对应12种地质灾害，子文件夹则是具体的子灾害类型（json）
# 请使用gpt-4o-mini模型进行样本扩增，调用方法见LLM_API.py
# 要求1、设计合适的提示词，英文
# 要求2、在指导时加入适当的示例，示例要包括原始人工撰写的样本两个（该种灾害该种问题人工编写的案例+其他灾害该种问题人工编写的案例（随机））
# 和LLM扩增的样本一个（该类灾害该种问题生成总数不小于3时再添加LLM生成的随机样本）
# json文件中每个样本的格式为{"instruction": "xxxx", "input": "xxxx", "output": "xxxx"}
# 其中instruction指明问题类型，You are a helpful geological disaster assistant. This is a {question} task.
# 包括question & answer, open-ended question, summary question, reasoning question四类问题
# 现在需要给每大类地质灾害的每种子灾害生成四类问题各10个样本，务必保持与人工编写的样本格式相同
# 扩增样本存放在 ./sft_samples/output/<disaster_type>/<sub_disaster_type>/ 目录下
import os
import json
import random
from LLM_API import LLM

class SampleAmplifier:
    def __init__(self):
        self.data_path = "../sft_samples/data"
        self.output_path = "../sft_samples/output"
        self.question_types = [
            "question & answer",
            "open-ended question", 
            "summary question",
            "reasoning question",
            
        ]
        self.target_samples_per_type = 10 # 每个类别、每个问题生成多少个样本
        self.generated_num = 3 # 生成多少个样本后开始使用扩增样本作为提示
        
    def load_existing_samples(self):
        """Load all existing samples from data directory"""
        all_samples = {}
        
        for disaster_type in os.listdir(self.data_path):
            disaster_path = os.path.join(self.data_path, disaster_type)
            if not os.path.isdir(disaster_path):
                continue
                
            all_samples[disaster_type] = {}
            
            for file_name in os.listdir(disaster_path):
                if file_name.endswith('.json'):
                    sub_disaster = file_name.replace('_en_en.json', '')
                    file_path = os.path.join(disaster_path, file_name)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        samples = json.load(f)
                        all_samples[disaster_type][sub_disaster] = samples
                        
        return all_samples
    
    def categorize_samples_by_type(self, samples):
        """Categorize samples by question type"""
        categorized = {qtype: [] for qtype in self.question_types}
        
        for sample in samples:
            instruction = sample['instruction']
            for qtype in self.question_types:
                if qtype in instruction:
                    categorized[qtype].append(sample)
                    break
                    
        return categorized
    
    def get_generated_samples_count(self, disaster_type, sub_disaster_type, question_type):
        """Count existing generated samples for a specific question type"""
        output_dir = os.path.join(self.output_path, disaster_type, sub_disaster_type)
        filename = f"{question_type.replace(' ', '_')}_generated.json"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                return len(samples)
        return 0
    
    def select_example_samples(self, all_samples, current_disaster, current_sub_disaster, question_type, generated_num):
        """Select example samples for prompt: 2 human-written + 1 LLM-generated (if available)"""
        examples = []
        
        # 第1个示例：来自当前正在生成的子灾害类型
        current_samples = all_samples[current_disaster][current_sub_disaster]
        current_categorized = self.categorize_samples_by_type(current_samples)
        current_type_samples = current_categorized[question_type]
        
        if current_type_samples:
            examples.append(random.choice(current_type_samples))
        
        # 第2个示例：从所有其他子灾害中等概率选择（包括其他灾害类型和当前灾害的其他子灾害）
        other_samples = []
        
        # 收集所有其他子灾害的样本
        for disaster, sub_disasters in all_samples.items():
            for sub_disaster, samples in sub_disasters.items():
                # 排除当前正在生成的子灾害
                if not (disaster == current_disaster and sub_disaster == current_sub_disaster):
                    categorized = self.categorize_samples_by_type(samples)
                    other_samples.extend(categorized[question_type])
        
        if other_samples:
            examples.append(random.choice(other_samples))
        
        # 第3个示例：LLM生成的样本（如果已生成数量 >= 3）
        generated_count = self.get_generated_samples_count(current_disaster, current_sub_disaster, question_type)
        if generated_count >= generated_num:
            output_dir = os.path.join(self.output_path, current_disaster, current_sub_disaster)
            filename = f"{question_type.replace(' ', '_')}_generated.json"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    generated_samples = json.load(f)
                    if generated_samples:
                        examples.append(random.choice(generated_samples))
            except Exception as e:
                print(f"    ⚠ Could not load LLM examples: {e}")
                
        return examples
    
    def create_prompt(self, disaster_type, sub_disaster_type, question_type, examples):
        """Create prompt for GPT-4o-mini"""
        
        prompt = f"""You are a professional geological disaster expert tasked with creating training samples for a geological disaster assistant. 

**Task**: Generate a high-quality sample for the disaster type "{disaster_type}" - "{sub_disaster_type}" with question type "{question_type}".

**Format Requirements**:
- instruction: "You are a helpful geological disaster assistant. This is a {question_type} task."
- input: A relevant question/request about {sub_disaster_type}
- output: A comprehensive, accurate, and professional response

**Question Type Guidelines**:
- question & answer: Direct factual questions requiring precise answers
- open-ended question: Questions allowing for detailed explanations and multiple perspectives  
- summary question: Questions requiring summarization of complex geological information
- reasoning question: Questions requiring analysis, cause-effect reasoning, or problem-solving

**Examples of similar samples**:
"""
        
        for i, example in enumerate(examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"instruction: {example['instruction']}\n"
            prompt += f"input: {example['input']}\n"
            prompt += f"output: {example['output']}\n"
        
        prompt += f"""

**Requirements**:
1. The content must be scientifically accurate and professionally written
2. Focus specifically on {sub_disaster_type} within the {disaster_type} category
3. Maintain the exact instruction format with "{question_type} task"
4. Provide professional, comprehensive, accurate, and non-fictional responses
5. Use proper geological terminology
6. Return only the JSON object with instruction, input, and output fields

Generate one sample now:"""
        print(prompt)
        return prompt
    
    def generate_single_sample(self, disaster_type, sub_disaster_type, question_type, examples):
        """Generate a single sample using GPT-4o-mini"""
        prompt = self.create_prompt(disaster_type, sub_disaster_type, question_type, examples)
        
        try:
            response = LLM("gpt-4o-mini", prompt)
            response_text = response.content.strip()
            
            # Extract JSON from response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            sample = json.loads(response_text)
            
            # Validate sample format
            if all(key in sample for key in ['instruction', 'input', 'output']):
                return sample
            else:
                print(f"✗ Invalid sample format")
                return None
                
        except Exception as e:
            print(f"✗ Error generating sample: {str(e)}")
            return None
    
    def append_sample_to_file(self, disaster_type, sub_disaster_type, question_type, sample):
        """Append a single sample to the output file"""
        output_dir = os.path.join(self.output_path, disaster_type, sub_disaster_type)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{question_type.replace(' ', '_')}_generated.json"
        filepath = os.path.join(output_dir, filename)
        
        # Load existing samples or create empty list
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                samples = json.load(f)
        else:
            samples = []
        
        # Append new sample
        samples.append(sample)
        
        # Save updated samples
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        return len(samples)
    
    def generate_samples_incrementally(self, all_samples, disaster_type, sub_disaster_type, question_type):
        """Generate samples incrementally, checking threshold for LLM examples"""
        current_count = self.get_generated_samples_count(disaster_type, sub_disaster_type, question_type)
        
        print(f"    Starting with {current_count} existing samples")
        
        for i in range(current_count, self.target_samples_per_type):
            print(f"    Generating sample {i+1}/{self.target_samples_per_type}")
            
            # Select examples (may include LLM examples if count >= 3)
            examples = self.select_example_samples(
                all_samples, disaster_type, sub_disaster_type, question_type, self.generated_num
            )
            
            if not examples:
                print(f"    ⚠ No examples found for {question_type}")
                break
            
            # Generate single sample
            sample = self.generate_single_sample(
                disaster_type, sub_disaster_type, question_type, examples
            )
            
            if sample:
                # Append to file immediately
                new_count = self.append_sample_to_file(
                    disaster_type, sub_disaster_type, question_type, sample
                )
                print(f"    ✓ Generated and saved sample {new_count}/{self.target_samples_per_type}")
            else:
                print(f"    ✗ Failed to generate sample {i+1}")
                
        final_count = self.get_generated_samples_count(disaster_type, sub_disaster_type, question_type)
        print(f"    Final count: {final_count}/{self.target_samples_per_type} samples")
    
    def amplify_all_samples(self):
        """Main function to amplify all samples"""
        print("Loading existing samples...")
        all_samples = self.load_existing_samples()
        
        print(f"Found {len(all_samples)} disaster types")
        
        for disaster_type, sub_disasters in all_samples.items():
            print(f"\nProcessing disaster type: {disaster_type}")
            
            for sub_disaster_type in sub_disasters.keys():
                print(f"  Processing sub-disaster: {sub_disaster_type}")
                
                for question_type in self.question_types:
                    print(f"    Processing question type: {question_type}")
                    
                    # Generate samples incrementally
                    self.generate_samples_incrementally(
                        all_samples, disaster_type, sub_disaster_type, question_type
                    )

def main():
    amplifier = SampleAmplifier()
    amplifier.amplify_all_samples()
    print("\nSample amplification completed!")

if __name__ == "__main__":
    main()

