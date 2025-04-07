import os
import pandas as pd
from pathlib import Path

import fitz  # PyMuPDF

cat_id_v2 = {
    "Material_didactico": 1,
    "Franqueros": 2,
    "Fonaindo": 3,
    "Actividad_Critica": 4,
}

def extract_pdf_text(pdf_path):
    """Extract text content from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def create_document_dataframe(root_directory):
    """
    Create a dataframe from PDF documents organized in category folders.
    
    Args:
        root_directory (str): Path to the directory containing category folders
        
    Returns:
        pandas.DataFrame: DataFrame with columns 'Eje Temático' and 'contenido'
    """
    data = []
    
    # Iterate through each category folder
    for category_folder in os.listdir(root_directory):
        category_path = os.path.join(root_directory, category_folder)
        
        # Skip if not a directory
        if not os.path.isdir(category_path):
            continue
        
        # Process each PDF in the category folder
        for file_name in os.listdir(category_path):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(category_path, file_name)
                
                # Extract content from PDF
                content = extract_pdf_text(file_path)
                
                # Add to data list
                data.append({
                    'EJE TEMÁTICO': cat_id_v2.get(category_folder),
                    'CONTENIDO': content
                })
    
    # Create DataFrame
    return pd.DataFrame(data)


def save_to_excel(df, output_path):
    """Save DataFrame to Excel file."""
    df.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")


def process_documents(root_directory, output_file='document_catalog.xlsx'):
    """
    Process all PDF documents in category folders and save to Excel.
    
    Args:
        root_directory (str): Path to the directory containing category folders
        output_file (str): Path to save the Excel output file
    """
    # Create dataframe from documents
    df = create_document_dataframe(root_directory)
    
    # Save to Excel
    save_to_excel(df, output_file)
    
    # Return summary
    return {
        'total_documents': len(df),
        'categories': df['EJE TEMÁTICO'].unique().tolist(),
        'output_file': output_file
    }


if __name__ == "__main__":
    # Example usage
    root_dir = "/Users/marioleiva/Documents/desarrollo/fairai/regex/v_2"
    result = process_documents(root_dir, "/Users/marioleiva/Documents/desarrollo/fairai/regex/v_2/categorias_documentos.xlsx")
    print(f"Processed {result['total_documents']} documents from {len(result['categories'])} categories.")