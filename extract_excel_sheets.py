import pandas as pd
from pathlib import Path

print("="*70)
print("Extracting NewsSumm Excel Sheets")
print("="*70)

# Paths
excel_file = Path('data/raw/newsumm/NewsSumm Dataset.xlsx')
output_path = Path('data/raw/newsumm')

print(f"\nReading: {excel_file.name}")

try:
    # Read Excel file and get all sheet names
    excel_data = pd.ExcelFile(excel_file)
    sheet_names = excel_data.sheet_names
    
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")
    
    # Process each sheet
    for sheet_name in sheet_names:
        print(f"\n📄 Processing sheet: {sheet_name}")
        
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Determine output filename based on sheet name
        sheet_lower = sheet_name.lower()
        
        if 'train' in sheet_lower:
            csv_name = 'train.csv'
        elif 'val' in sheet_lower or 'valid' in sheet_lower:
            csv_name = 'val.csv'
        elif 'test' in sheet_lower:
            csv_name = 'test.csv'
        else:
            # Use sheet name as filename
            csv_name = f'{sheet_name}.csv'
        
        # Save as CSV
        csv_file = output_path / csv_name
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"    Saved as: {csv_name}")
        
        # Show sample
        if len(df) > 0 and 'article' in df.columns:
            print(f"   First article preview: {str(df['article'].iloc[0])[:80]}...")
    
    print("\n" + "="*70)
    print(" All sheets extracted successfully!")
    print("="*70)
    
    # List created files
    print("\n Created CSV files:")
    csv_files = list(output_path.glob('*.csv'))
    for csv_file in csv_files:
        file_size = csv_file.stat().st_size / 1024 / 1024  # MB
        print(f"   {csv_file.name} ({file_size:.2f} MB)")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
