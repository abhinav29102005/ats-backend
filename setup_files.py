"""Setup all required files"""
import os

files_to_create = [
    'app/__init__.py',
    'app/core/__init__.py',
    'app/api/__init__.py',
    'app/utils/__init__.py',
]

for file_path in files_to_create:
    dir_path = os.path.dirname(file_path)
    
    # Create directory if doesn't exist
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"✅ Created directory: {dir_path}")
    
    # Create file if doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('"""\n')
            f.write(f'{os.path.basename(dir_path).title()} module\n')
            f.write('"""\n')
        print(f"✅ Created file: {file_path}")
    else:
        print(f"⏭️  Already exists: {file_path}")

print("\n✅ Setup complete! Try running your app now.")
