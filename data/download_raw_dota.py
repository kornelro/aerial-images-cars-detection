from src.data.download_raw_data import download
# import os
# import shutil


# main_dir = './data/raw/dota'
# if not os.path.exists(main_dir):
#     os.mkdir(main_dir)

# print('Downloading Dota images part 1...')
# download(
#     '1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2', main_dir,
#     unpack=False, zip_name='images.tar.001'
# )
# os.rename(main_dir+'/images', main_dir+'/part1')

# print('Downloading Dota images part 2...')
# download(
#     '1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v', main_dir,
#     unpack=False, zip_name='images.tar.002'
# )
# os.rename(main_dir+'/images', main_dir+'/part2')

# print('Downloading Dota images part 3...')
# download(
#     '1pEmwJtugIWhiwgBqOtplNUtTG2T454zn', main_dir,
#     unpack=False, zip_name='images.tar.003'
# )
# os.rename(main_dir+'/images', main_dir+'/part3')

# cat(
#     main_dir+'/images.tar',
#     main_dir+'/images.tar.001',
#     main_dir+'/images.tar.002',
#     main_dir+'/images.tar.003'
# )

# print('Downloading Dota images part 4...')
# download('1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP', main_dir)
# os.rename(main_dir+'/images', main_dir+'/part4')

# print('Downloading Dota annotations...')

# ann_dir = main_dir+'/ann1/'
# download('12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX', ann_dir)
# file_names = os.listdir(ann_dir)
# for file_name in file_names:
#     # TODO copy anotations only for files in folder
#     shutil.copy(os.path.join(ann_dir, file_name), main_dir+'/part1')
#     shutil.copy(os.path.join(ann_dir, file_name), main_dir+'/part2')
#     shutil.copy(os.path.join(ann_dir, file_name), main_dir+'/part3')
# shutil.rmtree(ann_dir)

# ann_dir = main_dir+'/ann2/'
# download('1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC', ann_dir)
# file_names = os.listdir(ann_dir)
# for file_name in file_names:
#     # TODO copy anotations only for files in folder
#     shutil.copy(os.path.join(ann_dir, file_name), main_dir+'/part4')
# shutil.rmtree(ann_dir)

print('Downloading DOTA...')
print('It\'ll take a while, take some coffee...')
download('1eGbKIiZ1-g_sAAZmK3OGAfP9OuFoQyh0', './data/raw/')