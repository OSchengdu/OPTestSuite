# OPTestSuite
The test frameworks under RISC-V platform and  ARM platform<br>
more details in atomgit `https://atomgit.com/polypopopo/operators`a

### **需求**  


#### **1. 核心需求分层**
| 层级 | 需求描述 | 技术挑战 | 技术适配 |
|------|----------|----------|--------------|
| **基础功能测试** | 验证算子数学正确性（如Conv2D输出误差<1e-6） | 多后端结果对比（PyTorch vs ONNX） | ✅ 已实现MXNet/PyTorch/ONNX自动化 |
| **性能基准测试** | 测量时延/吞吐量/功耗（支持国产GPU） | 硬件特异优化（如昇腾AI Core流水线） |  |
| **异常处理测试** | 注入NaN/INF/超大规模输入 | 内存泄漏检测（Valgrind/ASAN） |  |
| **分布式测试** | 多卡/多节点算子协同验证 | NCCL/RDMA通信优化 |  |
| **生产级部署** | CI/CD集成+可视化报表 | Prometheus+Grafana监控 |  |

