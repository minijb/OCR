from aip import AipOcr
 
appId= '19879132'
apiKey = 'UoQR3GOrB9el45lFQMG0dgGC'
secretKey = 'RA3SQ36wx4TLM7jaEHUcPVD6Dnp1Qv2M'

 
aipOcr  = AipOcr(appId, apiKey, secretKey)
filePath = "scan.jpg"
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 定义参数变量
options = {
  'detect_direction': 'true',
  'language_type': 'CHN_ENG',
}

# 调用通用文字识别接口
result = aipOcr.basicAccurate(get_file_content(filePath), options)


for item in result['words_result']:
    print(item['words'])
