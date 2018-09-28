# -*- coding=utf-8 -*-
import requests
from bs4 import BeautifulSoup


def get_result_ip(url, result_ip):
    s = requests.session()
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; "
                      "x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36"
    }
    rs = s.get(url=url, headers=header)
    soup = BeautifulSoup(rs.text, 'html5lib')
    ip_list_all = []
    ip_list = soup.select_one("#ip_list").select("tr")
    ip_info_list_key = ["ip", "port", "address", "hidden", "type", "speed", "conn_time", "survival_time", "verify_time"]

    for item in ip_list[1:]:
        ip_info_list_value = []
        ip_info = item.select("td")
        for info in ip_info[1:]:
            if info.select_one(".bar"):
                ip_info_list_value.append(info.select_one(".bar")["title"])
            else:
                ip_info_list_value.append(info.get_text().strip())
        ip_list_all.append(dict(zip(ip_info_list_key, ip_info_list_value)))
    for item in ip_list_all:
        speed = str(item['speed'])
        survival_time = str(item['survival_time'])
        speeds = float(speed[0:len(speed) - 1])
        if speeds < 1.0 and ('天' in survival_time or '小时' in survival_time):
            try:
                http_type = str(item['type']).lower()
                proxy_ip = http_type + "://" + item['ip'] + ":" + item['port']
                proxies = {http_type: proxy_ip}
                s.get('http://www.baidu.com', proxies=proxies, timeout=2)
            except:
                continue
            else:
                result_ip.append(proxies)


url_change = "http://www.xicidaili.com/nn/"
all_result_ip = []
for i in range(1, 10):
    get_result_ip(url_change + str(i), all_result_ip)
print(len(all_result_ip))


