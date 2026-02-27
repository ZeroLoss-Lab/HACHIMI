# API 配置文件夹

此文件夹用于存放 API 提供商的配置文件。

## 配置文件格式

创建 `providers.json` 文件，格式如下：

```json
{
  "providers": [
    {
      "name": "siliconflow",
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": "your-api-key-here",
      "model": "Qwen/Qwen2.5-72B-Instruct",
      "qpm": 60,
      "capacity_max": 300
    }
  ]
}
```

## 参数说明

- `name`: 提供商名称（自定义）
- `base_url`: API 基础 URL（OpenAI 协议兼容）
- `api_key`: 你的 API 密钥
- `model`: 模型名称
- `qpm`: 每分钟请求数配额
- `capacity_max`: 最大并发容量

## 多提供商配置

支持同时配置多个提供商进行负载均衡：

```json
{
  "providers": [
    {
      "name": "provider1",
      "base_url": "https://api.provider1.com/v1",
      "api_key": "key1",
      "model": "model1",
      "qpm": 60
    },
    {
      "name": "provider2", 
      "base_url": "https://api.provider2.com/v1",
      "api_key": "key2",
      "model": "model2",
      "qpm": 30
    }
  ]
}
```

## 注意事项

- 请勿将包含真实 API 密钥的 `providers.json` 提交到 Git 仓库
- 此文件夹已添加到 `.gitignore`
