import re
import csv
import os

def parse_log_file(log_file, version, region, data):
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Regex para encontrar todos os relatórios
    pattern = r"Performance counter stats for \'(.+?)\'(.*?seconds time elapsed\s+\(\s+\+\-\s+\d+\.\d+%\s+\))"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        command = match.group(1)
        report_content = match.group(2)
        
        # Extrair o tamanho de entrada do comando
        input_size = int(command.split()[-1])
        
        # Extrair as métricas do conteúdo do relatório
        context_match = re.search(r'(\d+(?:,\d+)?)\s+context-switches', report_content)
        page_faults_match = re.search(r'(\d+(?:,\d+)?)\s+page-faults', report_content)
        cycles_match = re.search(r'(\d+(?:,\d+)?)\s+cpu_core/cycles/', report_content)
        instructions_match = re.search(r'(\d+(?:,\d+)?)\s+cpu_core/instructions/', report_content)
        branches_match = re.search(r'(\d+(?:,\d+)?)\s+cpu_core/branches/', report_content)
        branch_misses_match = re.search(r'(\d+(?:,\d+)?)\s+cpu_core/branch-misses/', report_content)
        time_match = re.search(r'(\d+\.\d+)\s+seconds time elapsed\s+\(\s+\+\-\s+(\d+\.\d+)%\s+\)', report_content)
        
        # Substituir vírgulas e converter em inteiros ou floats
        context_switches = int(context_match.group(1).replace(',', '')) if context_match else None
        page_faults = int(page_faults_match.group(1).replace(',', '')) if page_faults_match else None
        cycles = int(cycles_match.group(1).replace(',', '')) if cycles_match else None
        instructions = int(instructions_match.group(1).replace(',', '')) if instructions_match else None
        branches = int(branches_match.group(1).replace(',', '')) if branches_match else None
        branch_misses = int(branch_misses_match.group(1).replace(',', '')) if branch_misses_match else None
        time = float(time_match.group(1)) if time_match else None
        time_stddev = float(time_match.group(2)) if time_match else None
        
        result = {
            "version": version,
            "region": region,
            "input_size": input_size,
            "context_switches": context_switches,
            "page_faults": page_faults,
            "cycles": cycles,
            "instructions": instructions,
            "branches": branches,
            "branch_misses": branch_misses,
            "time": time,
            "time_stddev": time_stddev
        }
        data.append(result) 

def write_to_csv(data, output_file='perf_stats.csv'):
    keys = data[0].keys()
    with open(output_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def process_logs(log_directory):
    data = []
    # Assumindo que os logs estejam organizados em diretórios por versão
    for version in ['mandelbrot_seq', 'mandelbrot_pth', 'mandelbrot_omp']:
        version_path = os.path.join(log_directory, version)
        
        for region in ['elephant', 'full', 'seahorse', 'triple_spiral']:
            log_file = os.path.join(version_path, f'{region}.log')
            if os.path.exists(log_file):
                log_data = parse_log_file(log_file, version, region, data)
            else:
                print(f"Log file {log_file} not found!")
    
    write_to_csv(data)

# Chamar a função com o diretório onde estão os logs
process_logs('../results/')
