#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests

refer_url = "http://hotel.qunar.com/city/xian/q-雁塔区"
ajax_url = "http://hotel.qunar.com/render/listPageSnapshot.jsp"
user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36"
seq = ['xian_13558','xian_32156','xian_21587']
data = {
    'seq': seq,
    'requestTime': '1534930073673',
    '__jscallback': 'XQScript_110'

}
headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Content-Length': '5651',
    'Content-Type': 'text/html;charset=UTF-8',
    'Host': 'hotel.qunar.com',
    'Origin': 'http://hotel.qunar.com',
    'Referer': refer_url,
    'User-Agent': user_agent,
    'X-Anit-Forge-Code': '0',
    'X-Anit-Forge-Token': 'None',
    'X-Requested-With': 'XMLHttpRequest',
}
resp = requests.post(ajax_url, data=data, headers=headers)
print(resp)

