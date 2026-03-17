import requests, json, time, os
from tqdm import tqdm

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def fetch_page(page_token=None, page_size=100):
    params = {
        "format": "json",
        "pageSize": page_size,
        "fields": "NCTId,EligibilityCriteria,BriefTitle,Condition,OverallStatus"
    }
    if page_token:
        params["pageToken"] = page_token
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def download_sample(max_records=500):
    """先下载少量数据验证流程，确认无误后再扩大规模"""
    all_studies = []
    next_token = None
    
    with tqdm(total=max_records, desc="Downloading") as pbar:
        while len(all_studies) < max_records:
            data = fetch_page(page_token=next_token)
            studies = data.get("studies", [])
            all_studies.extend(studies)
            pbar.update(len(studies))
            
            next_token = data.get("nextPageToken")
            if not next_token:
                break
            time.sleep(0.3)  # 避免请求过快
    
    # 保存原始数据
    out_path = os.path.join(OUTPUT_DIR, "raw_studies.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_studies[:max_records], f, ensure_ascii=False, indent=2)
    
    print(f"\n已下载 {len(all_studies[:max_records])} 条，保存到 {out_path}")
    return all_studies[:max_records]

if __name__ == "__main__":
    download_sample(max_records=500)