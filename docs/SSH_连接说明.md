# SSH 连接 Mac mini（Codex 用）

> 目标机器：Mac mini M5 Pro (64GB)，主机名 `SynCo-MacMini`
> IP：`192.168.3.13`
> 用户：`synco`
> 端口：`22`
> 本机配置完成日期：2026-11-20

---

## 一、连接信息

| 项目 | 值 |
|---|---|
| IP | `192.168.3.13` |
| 用户名 | `synco` |
| 端口 | `22` |
| SSH 别名（推荐） | `macmini` |
| 密钥文件 | `~/.ssh/id_ed25519` |
| 认证方式 | SSH 公钥（免密登录） |

---

## 二、如何连接（从本机）

### 方式 1：使用 SSH 别名（推荐，Codex 直接用这个）

```bash
ssh macmini
```

或执行远程命令：

```bash
ssh macmini 'uname -a'
ssh macmini 'ls ~'
```

### 方式 2：使用完整参数

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 synco@192.168.3.13
```

---

## 三、`~/.ssh/config` 内容

本机 `~/.ssh/config` 已写入：

```text
Host macmini
    HostName 192.168.3.13
    User synco
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```

如需连接其他机器（如 M4 48GB），在 `~/.ssh/config` 里追加对应条目即可。

---

## 四、密钥信息

- **类型**：Ed25519
- **私钥**：`~/.ssh/id_ed25519`（权限 600）
- **公钥**：`~/.ssh/id_ed25519.pub`
- **注释**：`augustus-codex-macmini`

**公钥内容（已追加到 Mac mini 的 `~/.ssh/authorized_keys`）：**

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL1P/OhP3fN73c8bvi7YCnav3LhHQJoolS4axrWzIcAf augustus-codex-macmini
```

---

## 五、测试命令（验证连接是否正常）

```bash
# 快速验证
ssh macmini 'echo LOGIN_OK && whoami'

# 查看硬件信息
ssh macmini 'system_profiler SPHardwareDataType | grep -E "Model Name|Chip|Cores|Memory"'

# 查看磁盘
ssh macmini 'df -h /'
```

---

## 六、Mac mini 硬件信息（已核实）

```text
Model Name: Mac mini
Chip:       Apple M5 Pro
Cores:      18 (6 Super + 12 Performance)
Memory:     64 GB
Disk:       926 GB SSD (18GB used, 67GB available)
OS:         macOS 27.0 (BuildVersion 26A425)
Host:       SynCo-MacMini
```

---

## 七、常用工作流

### 7.1 同步代码到 Mac mini

```bash
# 方式 1: rsync 整个仓库
rsync -avz --exclude='.git' \
  /Users/augustus/Desktop/LLMBenchmark/ \
  macmini:~/LLMBenchmark/

# 方式 2: 在 Mac mini 上 git clone（如果 Mac mini 能访问 GitHub）
ssh macmini 'git clone https://github.com/Augustus-Lee/LLMRouterBench.git ~/LLMBenchmark'
```

### 7.2 远程执行评测

```bash
ssh macmini 'cd ~/LLMBenchmark && source .venv-mlbench/bin/activate && python tools/mac_benchmark.py ...'
```

### 7.3 拉回评测结果

```bash
rsync -avz macmini:~/LLMBenchmark/results/ /Users/augustus/Desktop/LLMBenchmark/results/
```

---

## 八、故障排查

### 8.1 连接超时

```bash
# 1. ping
ping -c 2 192.168.3.13

# 2. 检查端口
nc -vz 192.168.3.13 22

# 3. 检查 Mac mini 是否开了远程登录
#    在 Mac mini 上: 系统设置 → 通用 → 共享 → 远程登录
```

### 8.2 权限被拒（publickey 失败）

```bash
# 用密码重新推送公钥
brew install hudochenkov/sshpass/sshpass  # 如未装
sshpass -p '<密码>' ssh-copy-id -i ~/.ssh/id_ed25519.pub synco@192.168.3.13

# 或手动
cat ~/.ssh/id_ed25519.pub | ssh synco@192.168.3.13 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

### 8.3 host key 变化

```bash
# 清除已知主机条目
ssh-keygen -R 192.168.3.13
# 再连一次
ssh macmini
```

---

## 九、安全说明

- 登录密码已在文档中脱敏，不记录具体值（仅首次安装公钥时使用过一次）
- 私钥只保存在本机 `~/.ssh/id_ed25519`，未上传任何仓库
- 推荐把 Mac mini 的 `PasswordAuthentication` 关掉，只允许公钥（`/etc/ssh/sshd_config`）
