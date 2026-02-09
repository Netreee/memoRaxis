
import json

def check_file():
    path = "out/acc_ret_results_13.json"
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        results = data.get("results", {})
        for adaptor, items in results.items():
            print(f"Checking {adaptor} with {len(items)} items...")
            for i, item in enumerate(items):
                if "answer" not in item:
                    print(f"MISSING 'answer' in {adaptor} at index {i}: {item}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_file()
