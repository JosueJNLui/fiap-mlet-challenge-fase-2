#!/usr/bin/env python
import os
import shutil
from pathlib import Path
import kagglehub

def main():
    # 1. Definir caminhos
    project_root = Path(__file__).resolve().parent.parent
    dest_dir = project_root / "data" / "raw"
    
    print("=== Iniciando Setup do Dataset MovieLens 20M ===")
    
    # 2. Fazer download via kagglehub
    print("Baixando dataset do Kaggle (via kagglehub)...")
    try:
        download_path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
        download_path = Path(download_path)
        print(f"Dataset baixado com sucesso em: {download_path}")
    except Exception as e:
        print(f"Erro ao baixar o dataset: {e}")
        return

    # 3. Criar diretório de destino local no repositório
    print(f"Criando diretório local de destino em: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copiar os arquivos para o repositório local
    print("Copiando arquivos do cache do kagglehub para o repositório local...")
    for item in download_path.iterdir():
        if item.is_file():
            dest_file = dest_dir / item.name
            # Evita recópias desnecessárias se o arquivo já existir e tiver o mesmo tamanho
            if dest_file.exists() and dest_file.stat().st_size == item.stat().st_size:
                print(f" - {item.name} já existe e está atualizado.")
            else:
                print(f" - Copiando {item.name}...")
                shutil.copy2(item, dest_file)
        elif item.is_dir():
            dest_subdir = dest_dir / item.name
            if dest_subdir.exists():
                shutil.rmtree(dest_subdir)
            print(f" - Copiando diretório {item.name}...")
            shutil.copytree(item, dest_subdir)

    print("\n=== Ingestão concluída com sucesso! ===")

if __name__ == "__main__":
    main()
