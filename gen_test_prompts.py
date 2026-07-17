import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path


def default_dataset_path():
    dataset_name = datetime.now().strftime("Dataset-%Y%m%d-%H%M")
    return Path(f"{dataset_name}.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the mixed prompt dataset and its visualization."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to "
            "Dataset-YYYYMMDD-HHMM.jsonl."
        ),
    )
    return parser.parse_args()


def save_dataset_artifacts(
    file_path,
    prompts,
    generation_metadata,
    save_visualization=True,
    visualization_output_dir=None,
    tokenizer_path=None,
):
    dataset_path = Path(file_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as dataset_file:
        for prompt in prompts:
            dataset_file.write(
                json.dumps(prompt, ensure_ascii=False) + "\n"
            )

    metadata_path = dataset_path.with_name(
        f"{dataset_path.stem}_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(generation_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "dataset": dataset_path,
        "metadata": metadata_path,
    }
    if save_visualization:
        from visualize_dataset import visualize_dataset

        output_dir = (
            Path(visualization_output_dir)
            if visualization_output_dir is not None
            else dataset_path.parent / "visualization"
        )
        chart_path, summary_path = visualize_dataset(
            dataset_path,
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
            generation_metadata=generation_metadata,
        )
        artifacts["chart"] = chart_path
        artifacts["summary"] = summary_path

    return artifacts


def generate_mixed_dataset(
    file_path,
    total_count,
    mean_percent,
    std_dev,
    num_long_prompts,
    output_mean_min=96,
    output_mean_max=384,
    output_curve_power=0.7,
    output_std_dev=48,
    output_min=64,
    output_max=512,
    output_seed=42,
    prompt_seed=42,
    save_visualization=True,
    visualization_output_dir=None,
    tokenizer_path=None,
):
    # 1. 生成Prompts
    p0 = "把下面内容翻译成英文。---\n\n# **2026年3月全球地缘政治与科技金融深度研报**  \n**——算力自主化、避险溢价与“十五五”战略的三重博弈**\n\n> **作者**：资深地缘政治战略家兼宏观经济首席分析师（20年一线实战经验）  \n> **时间**：2026年3月15日  \n> **核心逻辑**：全球科技—金融—地缘三重共振下，中国“十五五”开局之年正面临前所未有的“压力测试”。本报告以**算力自主化为矛、避险资产为盾、政策联动为轴**，穿透表象，揭示底层趋势与投资机会。\n\n---\n\n## **一、大国博弈与科技围堵：AI算力芯片制裁的“结构性反噬”与自主化加速**\n\n### **1.1 制裁动态：从“禁运”到“生态封锁”的质变**\n\n截至2026年3月，美国对华AI算力芯片的制裁已进入“全链条封锁”阶段。关键进展如下：\n\n- **2025年Q4起，美国商务部升级“实体清单”**，将**13家中国AI芯片设计公司**（含寒武纪、壁仞、摩尔线程）列入“先进算力设备出口限制名单”，禁止其获取**HBM3E（高带宽内存）** 与**先进封装技术**（如Chiplet、CoWoS）。\n- **2026年1月，美国联合荷兰ASML、日本东京电子、韩国三星**，签署《先进半导体供应链安全协议》（ASSA），限制**2nm以下制程设备**对华出口，且要求“**非美企业必须执行美国出口管制标准**”——即“长臂管辖”全球化。\n- **2026年2月，英伟达宣布“H20”芯片在华销售全面冻结**，其替代型号“H20-100”（专为非美市场设计）在实际性能上仅达H100的68%，且HBM3E带宽下降41%。\n\n> **关键数据对比**：\n> - 美国对华AI芯片出口额：2023年 $12.3B → 2025年 $4.1B → **2026年Q1仅 $1.8B**（同比下降77%）\n> - 中国HBM3E产能：2025年0.3万片/月 → 2026年3月**突破1.1万片/月**（中芯国际+长电科技+通富微电联合攻关）\n> - 全球AI算力密度（TOPS/W）：英伟达H100为**34.7 TOPS/W**，国产昇腾910B为**21.3 TOPS/W**，差距从2023年32%收窄至19%。\n\n### **1.2 反制逻辑：从“替代”到“重构”的算力自主化跃迁**\n\n制裁非但未遏制中国AI发展，反而**倒逼“算力调度优化+国产替代+生态重构”三位一体突破**：\n\n- **算力调度优化**：华为“盘古大模型”已实现**跨国产芯片异构算力池调度**，通过“算力资源池化+动态负载均衡”，在昇腾910B集群上实现**等效算力提升27%**（对比英伟达H100原生部署）。\n- **国产替代率跃升**：2026年Q1，中国AI服务器中**国产GPU占比达43%**（2023年仅12%），其中：\n  - 华为昇腾：占AI服务器市场28%\n  - 寒武纪思元370：在金融、政务领域市占率超15%\n  - 之江实验室“灵眸”芯片：已部署于杭州城市大脑，能效比超H100 12%\n- **生态重构加速**：百度“飞桨”、华为“MindSpore”、阿里“通义千问”已实现**全栈国产化AI训练框架**，支持昇腾/寒武纪/天数智芯等多厂商芯片。\n\n> **关键趋势**：  \n> 中国AI算力正从“**依赖进口芯片的算力堆叠**”转向“**国产芯片+异构调度+框架优化**”的**能效驱动型算力体系**。2026年，中国AI算力年增速预计达**41%**（全球平均28%），其中**国产算力贡献率超60%**。\n\n### **行动建议**  \n> **立即增持国产AI算力产业链核心标的**：优先配置**中芯国际（SMIC）+ 长电科技（JCET）+ 华为昇腾生态（如拓维信息、神州数码）**，并关注**HBM3E国产化供应链**（如江丰电子、沪硅产业）的突破性订单。**警惕英伟达H20替代品的“性能陷阱”**，避免误判国产芯片实际算力表现。\n\n---\n\n## **二、金融避险传导：地缘冲突溢价驱动黄金与A股避险板块的“双螺旋”共振**\n\n### **2.1 中东局势升级：地缘溢价的“通胀型”传导机制**\n\n2026年2月，**红海航运中断持续超67天**，沙特与伊朗在也门边境爆发新一轮空袭，以色列对黎巴嫩南部实施“定点清除”行动，引发全球供应链震荡。\n\n- **红海航运成本飙升**：2026年3月，从上海至鹿特丹的**集装箱运费达$18,500/FEU**（2023年平均$2,100），同比上涨781%。\n- **全球原油期货波动率（VIX-OIL）**：从2025年Q4的18.3升至2026年3月的**34.7**，逼近2022年俄乌冲突峰值。\n- **黄金价格**：2026年3月15日，**COMEX黄金期货报$2,342/盎司**，较2025年1月上涨**21.6%**，创历史新高。\n\n> **关键逻辑链**：  \n> **地缘冲突 → 航运中断 → 全球通胀预期升温 → 实际利率下行 → 黄金无息资产吸引力上升 → 资金流入黄金市场 → 黄金价格上行 → 通胀预期进一步强化 → 避险资产需求持续**\n\n### **2.2 避险传导机制：从黄金到A股的“三重传导路径”**\n\n黄金上涨并非孤立事件，其避险溢价正通过以下路径传导至A股：\n\n| 传导路径 | 机制 | 2026年Q1数据表现 |\n|--------|------|----------------|\n| **1. 资金再配置** | 外资与险资将黄金持仓比例提升至**11.8%**（2025年9.2%），部分资金回流A股避险板块 | 陆股通Q1净流入A股**1,280亿元**，其中能源+军工占比达63% |\n| **2. 通胀预期定价** | 大宗商品（原油、铜、镍）价格联动上涨，推动“资源类”公司利润预期上修 | 原油期货（WTI）3月均价$87.6/bbl（+23% YoY）；LME铜价$9,120/t（+18%） |\n| **3. 政策对冲预期** | 市场预期央行将启动“结构性降准+定向再贷款”对冲通胀风险，利好高分红、低估值板块 | A股股息率超5%的公司数量从2025年Q4的47家增至**89家** |\n\n> **关键数据**：  \n> - 2026年3月，**A股军工指数**（399967）较2025年1月上涨**38.4%**，**能源指数**（884021）上涨**29.7%**，均显著跑赢沪深300（+14.2%）。\n> - 黄金ETF（518880）3月单月资金净流入**32.7亿元**，为2024年以来最高。\n\n绝地求生：四川航空3U8633航班“5·14”特大险情全景纪实\n\n第一部分：航班与背景信息\n\n1. 航班基本信息\n* 航班号：四川航空 3U8633（飞行呼号：Sichuan 8633）\n* 日期：2018年5月14日\n* 航线：重庆江北国际机场（CKG） 飞往 拉萨贡嘎国际机场（LXA）\n* 机型：空中客车 A319-133（注册号 B-6419）。该机型被称为“高原雄鹰”，专为高海拔飞行设计，具有优异的高原性能。\n* 人员：机上共载有128人，其中旅客119名，机组人员9名（含3名飞行员、5名乘务员、1名安全员）。\n\n2. 核心机组人员（“英雄机组”）\n* 责任机长：刘传健。前空军轰炸机飞行员，拥有极高的飞行天赋和丰富的藏区航线飞行经验，心理素质极其过硬。\n* 第二机长：梁鹏。当天作为带飞教员坐在观察员位置，在事故发生后起到了至关重要的辅助和通讯作用。\n* 副驾驶：徐瑞辰。事故发生时坐在右座，直接承受了风挡破裂带来的巨大物理伤害。\n* 乘务长：毕楠。负责客舱安全与秩序，在极度恐慌中稳定了全体乘客的情绪。\n\n第二部分：平静的巡航与瞬间的灾难（06:25 - 07:07）\n\n1. 正常的起飞与爬升\n清晨06:25，3U8633航班从重庆江北机场顺利起飞。航线前方是著名的青藏高原，这也是世界上飞行难度最高的区域之一，地形复杂，气候多变。\n飞机平稳爬升至9800米（约32100英尺）的巡航高度，进入成都区域。此时自动驾驶仪（AP）接通，机组人员按照标准程序进行巡航监控，客舱内的乘务员开始为旅客分发早餐，一切显得平静而日常。\n\n2. 毫无征兆的裂纹\n07:07左右，在距离成都约100公里处的空域，驾驶舱内突然传来“嘭”的一声闷响。\n机长刘传健敏锐地发现，右座（副驾驶一侧）的前风挡玻璃上出现了网状裂纹。在万米高空，风挡玻璃出现裂纹是极其危险的信号。刘传健立刻伸手触摸裂纹，判断出是内层玻璃破裂。\n他没有丝毫犹豫，立即通过无线电向西南空管局成都区域管制中心（ATC）报告：\n“成都，四川8633。”\n“请讲。”\n“我右边风挡裂了，我申请下高度，返航。”\n“四川8633，下高度8400，保持当前航向进入成都。”\n\n3. 爆炸性失压\n就在刘传健与管制员结束通话后不到一秒钟，伴随着一声震耳欲聋的巨响，右侧风挡玻璃完全爆裂并脱落飞出机外。\n瞬间，驾驶舱内发生了爆炸性失压（Explosive Decompression）。机舱内原本被加压的空气如同决堤的洪水一般，疯狂向外泄露。\n\n第三部分：万米高空的“绝境”与生理极限\n\n风挡脱落后，驾驶舱瞬间变成了地狱般的恶劣环境。机长刘传健和副驾驶徐瑞辰面临着多重致命威胁：\n\n1. 极寒与狂风\n飞机当时的飞行速度约为800公里/小时（接近0.75马赫）。挡风玻璃消失后，零下40摄氏度（-40℃）的刺骨寒风以狂暴的速度直接灌入驾驶舱。机组人员当时仅穿着短袖制服，瞬间处于极度失温的状态。强风让刘传健几乎无法睁开眼睛，面部肌肉被风吹得变形。\n\n2. 极度缺氧\n9800米高空的氧气含量仅为海平面的不到三分之一。在没有氧气面罩的情况下，人类在这一高度的“有效意识时间”（TUC）仅有几十秒到一分钟。一旦失去意识，整架飞机将陷入万劫不复的境地。\n\n3. 副驾驶被吸出窗外\n由于压差巨大，副驾驶徐瑞辰的半个身体瞬间被吸出了窗外。狂风撕裂了他的制服，幸好他系着安全带（大腿根部的安全带将他拉住），才没有完全飞出机外。在接下来的飞行中，强风又将他“拍”回了座椅上，但他已受了伤，且在寒风中冻得瑟瑟发抖，无法进行任何操作。\n\n4. 震耳欲聋的噪音\n狂风灌入驾驶舱产生了超过100分贝的巨大噪音，就如同有人在耳边持续开枪。机长和副驾驶之间完全无法通过语言交流，也根本听不到无线电里地面管制员的呼叫。\n\n5. 仪表盘损毁，自动驾驶失效\n爆炸性的强风将飞行控制面板（FCU）吹翻，驾驶舱内许多电子设备失灵，警报声大作。更致命的是，自动驾驶系统断开，飞机开始出现剧烈的颠簸和姿态倾斜。\n\n第四部分：史诗级的盲飞与迫降（07:08 - 07:46）\n\n在这个连呼吸都困难的绝境中，机长刘传健展现出了人类顶级的职业素养和心理承受力。\n\n1. 肌肉记忆与艰难控机\n在极寒、缺氧、闭眼的情况下，刘传健完全凭借着几十年的飞行经验和“肌肉记忆”抓住了操纵杆。他必须先将飞机的姿态改平，防止飞机失控坠毁。在强大的气流阻力下，操纵杆变得异常沉重，每一次拉杆都需要耗尽全身的力气。\n\n2. 不能立刻下降的生死抉择\n按照标准的失压处置程序，飞机应该立即紧急下降到3000米（约10000英尺）的安全高度，以获取足够的氧气和适宜的温度。\n但是，3U8633当时正飞行在青藏高原东部边缘（青康藏高原）的崇山峻岭之上。下方的山脉海拔普遍在五六千米以上。如果盲目下降到3000米，飞机将直接撞山。\n刘传健清醒地意识到这一点，他强忍着缺氧和严寒，控制飞机先以一个相对平缓的下降率下降到7300米（约24000英尺）——这是该区域的最低安全高度（MORA），在这个高度上艰难平飞，等待飞越山脉。\n\n3. 第二机长进入驾驶舱\n此时，原本在客舱休息的第二机长梁鹏察觉到了异常。他顶着巨大的狂风和客舱失压带来的不适，艰难地推开驾驶舱门进入。\n梁鹏的加入起到了关键作用。他立刻为刘传健戴上氧气面罩（刘传健因双手必须死死控制操纵杆，无法自己佩戴），随后在极度嘈杂的环境中，梁鹏通过极其艰难的沟通，协助刘传健进行导航规划，并不断按摩刘传健冻僵的手臂，帮他保持体温。同时，梁鹏在盲音中向管制发出“MAYDAY MAYDAY”（国际通用最高级别求救信号）并在应答机上挂出了7700的紧急代码。\n\n4. 越过高山，超重降落\n在艰难熬过了十几分钟的“死亡平飞”后，飞机终于飞出了山区，进入了四川盆地。刘传健立刻操纵飞机大幅度下降高度，气温和氧气状况开始好转。\n此时，新的问题出现了：\n由于飞机刚起飞不久，油箱里几乎装满了前往拉萨的高原燃油，飞机处于严重超重状态。A319机型不具备空中放油功能，机组也根本没有时间进行盘旋耗油。\n带着几十吨未消耗的燃油降落，极易导致起落架折断、轮胎爆裂甚至冲出跑道起火。但在人命关天的时刻，刘传健别无选择。\n\n07:46，凭借极其精湛的技术，刘传健驾驶着满目疮痍的3U8633航班，以极其平稳的姿态，超重降落在成都双流国际机场的跑道上。\n由于超重和刹车过猛，起落架轮胎温度急剧升高，热熔塞自动熔断放气（轮胎变瘪），飞机稳稳地停在了跑道上。\n\n全机119名乘客和9名机组人员，全部生还。\n\n第五部分：客舱的恐慌与地面的营救\n\n1. 客舱内的“生死15分钟”\n当风挡脱落的瞬间，客舱同样经历了猛烈的失压。由于客舱空气被吸出，气温骤降，氧气面罩自动脱落。\n飞机在万米高空发生剧烈颠簸并伴有失重感，客舱内杂物乱飞，乘客陷入了极度的恐慌，有人开始哭泣，甚至有人开始写遗书。\n此时，乘务长毕楠和全体乘务组展现了极高的专业精神。虽然她们也感到恐惧，甚至有人在剧烈颠簸中摔倒受轻伤，但她们迅速戴上氧气面罩，通过大喇叭和声嘶力竭的吼叫，安抚乘客：\n“请大家相信我们！相信我们有能力带领大家安全落地！”\n“低下头，系好安全带！不要慌张！”\n乘务员们逐一检查乘客的安全带和氧气面罩，在混乱中建立起了秩序，避免了因恐慌导致的二次伤害。\n\n2. 地面管制的“清空行动”\n当雷达屏幕上3U8633航班的高度骤降，且应答机闪烁着7700的红光时，成都区管中心立刻进入了最高级别的应急响应。\n管制员在无线电中一遍遍盲呼：“四川8633，成都叫你。”但始终得不到回应。\n考虑到飞机可能已经失控或者正在紧急迫降，地面空管做出了最果断的决定：清空空域。\n所有在成都附近飞行的航班都被要求避让，进近航班全部复飞，成都双流机场跑道全部清空，消防车、救护车全部在跑道两侧就位，为3U8633敞开了一条没有任何阻碍的“生命通道”。\n\n第六部分：官方调查报告与事故原因剖析\n\n2020年6月，中国民用航空局（CAAC）正式发布了《四川航空A319-100/B-6419号机“5·14”重大飞行事故调查报告》。这揭示了灾难发生的根本机械原因：\n\n1. 封严破损与水汽渗入：涉事飞机的右侧风挡玻璃边缘的封严硅胶出现了微小的破损。在长期的飞行中，外部环境的水汽通过这个破损处，渗入并存留在风挡玻璃底部的空腔内。\n2. 绝缘层性能下降：风挡玻璃内部装有用于防冰防雾的电加热系统。水汽的长期浸泡，导致加热系统底部导线的绝缘材料性能严重下降。\n3. 电弧放电（Arcing）：在事发当天，绝缘性能进一步恶化，导致导线之间或导线与周围结构之间产生了持续的电弧放电。\n4. 局部高温爆裂：电弧放电产生了极高温度的局部热点。风挡玻璃属于多层结构，受热极度不均，导致双层结构玻璃瞬间发生破裂。\n5. 无法承受压差脱落：玻璃结构被热应力破坏后，彻底丧失了结构强度，无法承受万米高空驾驶舱内外巨大的压力差，最终导致整块风挡玻璃从机身上爆裂脱落。\n\n调查报告明确指出，这是一次由极其隐蔽的机械缺陷引发的突发事故，在当时的常规航线维修检查中，是几乎无法通过目视检查提前发现的。\n\n第七部分：深远影响与历史意义\n\n1. 航空史上的奇迹：这次备降被世界航空界公认为“史诗级”的操作，难度甚至超越了著名的“哈德逊河奇迹”（全美航空1549号航班）。刘传健机长被授予“中国民航英雄机长”称号，3U8633机组被授予“中国民航英雄机组”称号。\n2. 促进行业安全升级：事故发生后，空客公司发布了针对A320系列飞机风挡玻璃的紧急服务通告，全球各大航空公司修改了维修手册，增加了对风挡玻璃加热系统绝缘电阻的定期测试，从源头上堵住了这一潜在的安全漏洞。\n3. 文化记忆：这一真实事件被改编为电影《中国机长》（2019年上映），让更多公众了解了民航人的不易与坚守，成为了中国民航安全文化的一座丰碑。"
    p0 = p0 * 20
    ratio = generate_ratio_list(
        total_count,
        mean_percent,
        std_dev,
        num_long_prompts,
        seed=prompt_seed,
    )

    # 2. 排列Prompts
    prompts = gen_prompts(
        ratio,
        p0,
        output_mean_min=output_mean_min,
        output_mean_max=output_mean_max,
        output_curve_power=output_curve_power,
        output_std_dev=output_std_dev,
        output_min=output_min,
        output_max=output_max,
        output_seed=output_seed,
    )
    
    generation_metadata = {
        "total_count": total_count,
        "prompt_distribution": {
            "type": "normal_ratio",
            "formula": "R ~ Normal(mean, std_dev); R = clip(R, 0, 1)",
            "mean": mean_percent / 100.0,
            "std_dev": std_dev,
            "num_long_prompts": num_long_prompts,
            "seed": prompt_seed,
        },
        "output_distribution": {
            "type": "conditional_truncated_normal",
            "formula": (
                "r_i=L_i/L_max; "
                "mu_i=mu_min+(mu_max-mu_min)*r_i^power; "
                "O_i~TruncNormal(mu_i, std_dev, min<=O_i<=max)"
            ),
            "mean_min": output_mean_min,
            "mean_max": output_mean_max,
            "curve_power": output_curve_power,
            "std_dev": output_std_dev,
            "min": output_min,
            "max": output_max,
            "seed": output_seed,
        },
    }
    return save_dataset_artifacts(
        file_path,
        prompts,
        generation_metadata,
        save_visualization=save_visualization,
        visualization_output_dir=visualization_output_dir,
        tokenizer_path=tokenizer_path,
    )

def generate_mixed_dataset_lognormal(
    file_path,
    total_count,
    mu,
    sigma,
    num_long_prompts,
    output_mean_min=96,
    output_mean_max=384,
    output_curve_power=0.7,
    output_std_dev=48,
    output_min=64,
    output_max=512,
    output_seed=42,
    prompt_seed=42,
    save_visualization=True,
    visualization_output_dir=None,
    tokenizer_path=None,
):
    # 1. 生成基础长文本
    p0 = "把下面内容翻译成英文。---\n\n# **2026年3月全球地缘政治与科技金融深度研报**  \n**——算力自主化、避险溢价与“十五五”战略的三重博弈**\n\n> **作者**：资深地缘政治战略家兼宏观经济首席分析师（20年一线实战经验）  \n> **时间**：2026年3月15日  \n> **核心逻辑**：全球科技—金融—地缘三重共振下，中国“十五五”开局之年正面临前所未有的“压力测试”。本报告以**算力自主化为矛、避险资产为盾、政策联动为轴**，穿透表象，揭示底层趋势与投资机会。\n\n---\n\n## **一、大国博弈与科技围堵：AI算力芯片制裁的“结构性反噬”与自主化加速**\n\n### **1.1 制裁动态：从“禁运”到“生态封锁”的质变**\n\n截至2026年3月，美国对华AI算力芯片的制裁已进入“全链条封锁”阶段。关键进展如下：\n\n- **2025年Q4起，美国商务部升级“实体清单”**，将**13家中国AI芯片设计公司**（含寒武纪、壁仞、摩尔线程）列入“先进算力设备出口限制名单”，禁止其获取**HBM3E（高带宽内存）** 与**先进封装技术**（如Chiplet、CoWoS）。\n- **2026年1月，美国联合荷兰ASML、日本东京电子、韩国三星**，签署《先进半导体供应链安全协议》（ASSA），限制**2nm以下制程设备**对华出口，且要求“**非美企业必须执行美国出口管制标准**”——即“长臂管辖”全球化。\n- **2026年2月，英伟达宣布“H20”芯片在华销售全面冻结**，其替代型号“H20-100”（专为非美市场设计）在实际性能上仅达H100的68%，且HBM3E带宽下降41%。\n\n> **关键数据对比**：\n> - 美国对华AI芯片出口额：2023年 $12.3B → 2025年 $4.1B → **2026年Q1仅 $1.8B**（同比下降77%）\n> - 中国HBM3E产能：2025年0.3万片/月 → 2026年3月**突破1.1万片/月**（中芯国际+长电科技+通富微电联合攻关）\n> - 全球AI算力密度（TOPS/W）：英伟达H100为**34.7 TOPS/W**，国产昇腾910B为**21.3 TOPS/W**，差距从2023年32%收窄至19%。\n\n### **1.2 反制逻辑：从“替代”到“重构”的算力自主化跃迁**\n\n制裁非但未遏制中国AI发展，反而**倒逼“算力调度优化+国产替代+生态重构”三位一体突破**：\n\n- **算力调度优化**：华为“盘古大模型”已实现**跨国产芯片异构算力池调度**，通过“算力资源池化+动态负载均衡”，在昇腾910B集群上实现**等效算力提升27%**（对比英伟达H100原生部署）。\n- **国产替代率跃升**：2026年Q1，中国AI服务器中**国产GPU占比达43%**（2023年仅12%），其中：\n  - 华为昇腾：占AI服务器市场28%\n  - 寒武纪思元370：在金融、政务领域市占率超15%\n  - 之江实验室“灵眸”芯片：已部署于杭州城市大脑，能效比超H100 12%\n- **生态重构加速**：百度“飞桨”、华为“MindSpore”、阿里“通义千问”已实现**全栈国产化AI训练框架**，支持昇腾/寒武纪/天数智芯等多厂商芯片。\n\n> **关键趋势**：  \n> 中国AI算力正从“**依赖进口芯片的算力堆叠**”转向“**国产芯片+异构调度+框架优化**”的**能效驱动型算力体系**。2026年，中国AI算力年增速预计达**41%**（全球平均28%），其中**国产算力贡献率超60%**。\n\n### **行动建议**  \n> **立即增持国产AI算力产业链核心标的**：优先配置**中芯国际（SMIC）+ 长电科技（JCET）+ 华为昇腾生态（如拓维信息、神州数码）**，并关注**HBM3E国产化供应链**（如江丰电子、沪硅产业）的突破性订单。**警惕英伟达H20替代品的“性能陷阱”**，避免误判国产芯片实际算力表现。\n\n---\n\n## **二、金融避险传导：地缘冲突溢价驱动黄金与A股避险板块的“双螺旋”共振**\n\n### **2.1 中东局势升级：地缘溢价的“通胀型”传导机制**\n\n2026年2月，**红海航运中断持续超67天**，沙特与伊朗在也门边境爆发新一轮空袭，以色列对黎巴嫩南部实施“定点清除”行动，引发全球供应链震荡。\n\n- **红海航运成本飙升**：2026年3月，从上海至鹿特丹的**集装箱运费达$18,500/FEU**（2023年平均$2,100），同比上涨781%。\n- **全球原油期货波动率（VIX-OIL）**：从2025年Q4的18.3升至2026年3月的**34.7**，逼近2022年俄乌冲突峰值。\n- **黄金价格**：2026年3月15日，**COMEX黄金期货报$2,342/盎司**，较2025年1月上涨**21.6%**，创历史新高。\n\n> **关键逻辑链**：  \n> **地缘冲突 → 航运中断 → 全球通胀预期升温 → 实际利率下行 → 黄金无息资产吸引力上升 → 资金流入黄金市场 → 黄金价格上行 → 通胀预期进一步强化 → 避险资产需求持续**\n\n### **2.2 避险传导机制：从黄金到A股的“三重传导路径”**\n\n黄金上涨并非孤立事件，其避险溢价正通过以下路径传导至A股：\n\n| 传导路径 | 机制 | 2026年Q1数据表现 |\n|--------|------|----------------|\n| **1. 资金再配置** | 外资与险资将黄金持仓比例提升至**11.8%**（2025年9.2%），部分资金回流A股避险板块 | 陆股通Q1净流入A股**1,280亿元**，其中能源+军工占比达63% |\n| **2. 通胀预期定价** | 大宗商品（原油、铜、镍）价格联动上涨，推动“资源类”公司利润预期上修 | 原油期货（WTI）3月均价$87.6/bbl（+23% YoY）；LME铜价$9,120/t（+18%） |\n| **3. 政策对冲预期** | 市场预期央行将启动“结构性降准+定向再贷款”对冲通胀风险，利好高分红、低估值板块 | A股股息率超5%的公司数量从2025年Q4的47家增至**89家** |\n\n> **关键数据**：  \n> - 2026年3月，**A股军工指数**（399967）较2025年1月上涨**38.4%**，**能源指数**（884021）上涨**29.7%**，均显著跑赢沪深300（+14.2%）。\n> - 黄金ETF（518880）3月单月资金净流入**32.7亿元**，为2024年以来最高。\n\n绝地求生：四川航空3U8633航班“5·14”特大险情全景纪实\n\n第一部分：航班与背景信息\n\n1. 航班基本信息\n* 航班号：四川航空 3U8633（飞行呼号：Sichuan 8633）\n* 日期：2018年5月14日\n* 航线：重庆江北国际机场（CKG） 飞往 拉萨贡嘎国际机场（LXA）\n* 机型：空中客车 A319-133（注册号 B-6419）。该机型被称为“高原雄鹰”，专为高海拔飞行设计，具有优异的高原性能。\n* 人员：机上共载有128人，其中旅客119名，机组人员9名（含3名飞行员、5名乘务员、1名安全员）。\n\n2. 核心机组人员（“英雄机组”）\n* 责任机长：刘传健。前空军轰炸机飞行员，拥有极高的飞行天赋和丰富的藏区航线飞行经验，心理素质极其过硬。\n* 第二机长：梁鹏。当天作为带飞教员坐在观察员位置，在事故发生后起到了至关重要的辅助和通讯作用。\n* 副驾驶：徐瑞辰。事故发生时坐在右座，直接承受了风挡破裂带来的巨大物理伤害。\n* 乘务长：毕楠。负责客舱安全与秩序，在极度恐慌中稳定了全体乘客的情绪。\n\n第二部分：平静的巡航与瞬间的灾难（06:25 - 07:07）\n\n1. 正常的起飞与爬升\n清晨06:25，3U8633航班从重庆江北机场顺利起飞。航线前方是著名的青藏高原，这也是世界上飞行难度最高的区域之一，地形复杂，气候多变。\n飞机平稳爬升至9800米（约32100英尺）的巡航高度，进入成都区域。此时自动驾驶仪（AP）接通，机组人员按照标准程序进行巡航监控，客舱内的乘务员开始为旅客分发早餐，一切显得平静而日常。\n\n2. 毫无征兆的裂纹\n07:07左右，在距离成都约100公里处的空域，驾驶舱内突然传来“嘭”的一声闷响。\n机长刘传健敏锐地发现，右座（副驾驶一侧）的前风挡玻璃上出现了网状裂纹。在万米高空，风挡玻璃出现裂纹是极其危险的信号。刘传健立刻伸手触摸裂纹，判断出是内层玻璃破裂。\n他没有丝毫犹豫，立即通过无线电向西南空管局成都区域管制中心（ATC）报告：\n“成都，四川8633。”\n“请讲。”\n“我右边风挡裂了，我申请下高度，返航。”\n“四川8633，下高度8400，保持当前航向进入成都。”\n\n3. 爆炸性失压\n就在刘传健与管制员结束通话后不到一秒钟，伴随着一声震耳欲聋的巨响，右侧风挡玻璃完全爆裂并脱落飞出机外。\n瞬间，驾驶舱内发生了爆炸性失压（Explosive Decompression）。机舱内原本被加压的空气如同决堤的洪水一般，疯狂向外泄露。\n\n第三部分：万米高空的“绝境”与生理极限\n\n风挡脱落后，驾驶舱瞬间变成了地狱般的恶劣环境。机长刘传健和副驾驶徐瑞辰面临着多重致命威胁：\n\n1. 极寒与狂风\n飞机当时的飞行速度约为800公里/小时（接近0.75马赫）。挡风玻璃消失后，零下40摄氏度（-40℃）的刺骨寒风以狂暴的速度直接灌入驾驶舱。机组人员当时仅穿着短袖制服，瞬间处于极度失温的状态。强风让刘传健几乎无法睁开眼睛，面部肌肉被风吹得变形。\n\n2. 极度缺氧\n9800米高空的氧气含量仅为海平面的不到三分之一。在没有氧气面罩的情况下，人类在这一高度的“有效意识时间”（TUC）仅有几十秒到一分钟。一旦失去意识，整架飞机将陷入万劫不复的境地。\n\n3. 副驾驶被吸出窗外\n由于压差巨大，副驾驶徐瑞辰的半个身体瞬间被吸出了窗外。狂风撕裂了他的制服，幸好他系着安全带（大腿根部的安全带将他拉住），才没有完全飞出机外。在接下来的飞行中，强风又将他“拍”回了座椅上，但他已受了伤，且在寒风中冻得瑟瑟发抖，无法进行任何操作。\n\n4. 震耳欲聋的噪音\n狂风灌入驾驶舱产生了超过100分贝的巨大噪音，就如同有人在耳边持续开枪。机长和副驾驶之间完全无法通过语言交流，也根本听不到无线电里地面管制员的呼叫。\n\n5. 仪表盘损毁，自动驾驶失效\n爆炸性的强风将飞行控制面板（FCU）吹翻，驾驶舱内许多电子设备失灵，警报声大作。更致命的是，自动驾驶系统断开，飞机开始出现剧烈的颠簸和姿态倾斜。\n\n第四部分：史诗级的盲飞与迫降（07:08 - 07:46）\n\n在这个连呼吸都困难的绝境中，机长刘传健展现出了人类顶级的职业素养和心理承受力。\n\n1. 肌肉记忆与艰难控机\n在极寒、缺氧、闭眼的情况下，刘传健完全凭借着几十年的飞行经验和“肌肉记忆”抓住了操纵杆。他必须先将飞机的姿态改平，防止飞机失控坠毁。在强大的气流阻力下，操纵杆变得异常沉重，每一次拉杆都需要耗尽全身的力气。\n\n2. 不能立刻下降的生死抉择\n按照标准的失压处置程序，飞机应该立即紧急下降到3000米（约10000英尺）的安全高度，以获取足够的氧气和适宜的温度。\n但是，3U8633当时正飞行在青藏高原东部边缘（青康藏高原）的崇山峻岭之上。下方的山脉海拔普遍在五六千米以上。如果盲目下降到3000米，飞机将直接撞山。\n刘传健清醒地意识到这一点，他强忍着缺氧和严寒，控制飞机先以一个相对平缓的下降率下降到7300米（约24000英尺）——这是该区域的最低安全高度（MORA），在这个高度上艰难平飞，等待飞越山脉。\n\n3. 第二机长进入驾驶舱\n此时，原本在客舱休息的第二机长梁鹏察觉到了异常。他顶着巨大的狂风和客舱失压带来的不适，艰难地推开驾驶舱门进入。\n梁鹏的加入起到了关键作用。他立刻为刘传健戴上氧气面罩（刘传健因双手必须死死控制操纵杆，无法自己佩戴），随后在极度嘈杂的环境中，梁鹏通过极其艰难的沟通，协助刘传健进行导航规划，并不断按摩刘传健冻僵的手臂，帮他保持体温。同时，梁鹏在盲音中向管制发出“MAYDAY MAYDAY”（国际通用最高级别求救信号）并在应答机上挂出了7700的紧急代码。\n\n4. 越过高山，超重降落\n在艰难熬过了十几分钟的“死亡平飞”后，飞机终于飞出了山区，进入了四川盆地。刘传健立刻操纵飞机大幅度下降高度，气温和氧气状况开始好转。\n此时，新的问题出现了：\n由于飞机刚起飞不久，油箱里几乎装满了前往拉萨的高原燃油，飞机处于严重超重状态。A319机型不具备空中放油功能，机组也根本没有时间进行盘旋耗油。\n带着几十吨未消耗的燃油降落，极易导致起落架折断、轮胎爆裂甚至冲出跑道起火。但在人命关天的时刻，刘传健别无选择。\n\n07:46，凭借极其精湛的技术，刘传健驾驶着满目疮痍的3U8633航班，以极其平稳的姿态，超重降落在成都双流国际机场的跑道上。\n由于超重和刹车过猛，起落架轮胎温度急剧升高，热熔塞自动熔断放气（轮胎变瘪），飞机稳稳地停在了跑道上。\n\n全机119名乘客和9名机组人员，全部生还。\n\n第五部分：客舱的恐慌与地面的营救\n\n1. 客舱内的“生死15分钟”\n当风挡脱落的瞬间，客舱同样经历了猛烈的失压。由于客舱空气被吸出，气温骤降，氧气面罩自动脱落。\n飞机在万米高空发生剧烈颠簸并伴有失重感，客舱内杂物乱飞，乘客陷入了极度的恐慌，有人开始哭泣，甚至有人开始写遗书。\n此时，乘务长毕楠和全体乘务组展现了极高的专业精神。虽然她们也感到恐惧，甚至有人在剧烈颠簸中摔倒受轻伤，但她们迅速戴上氧气面罩，通过大喇叭和声嘶力竭的吼叫，安抚乘客：\n“请大家相信我们！相信我们有能力带领大家安全落地！”\n“低下头，系好安全带！不要慌张！”\n乘务员们逐一检查乘客的安全带和氧气面罩，在混乱中建立起了秩序，避免了因恐慌导致的二次伤害。\n\n2. 地面管制的“清空行动”\n当雷达屏幕上3U8633航班的高度骤降，且应答机闪烁着7700的红光时，成都区管中心立刻进入了最高级别的应急响应。\n管制员在无线电中一遍遍盲呼：“四川8633，成都叫你。”但始终得不到回应。\n考虑到飞机可能已经失控或者正在紧急迫降，地面空管做出了最果断的决定：清空空域。\n所有在成都附近飞行的航班都被要求避让，进近航班全部复飞，成都双流机场跑道全部清空，消防车、救护车全部在跑道两侧就位，为3U8633敞开了一条没有任何阻碍的“生命通道”。\n\n第六部分：官方调查报告与事故原因剖析\n\n2020年6月，中国民用航空局（CAAC）正式发布了《四川航空A319-100/B-6419号机“5·14”重大飞行事故调查报告》。这揭示了灾难发生的根本机械原因：\n\n1. 封严破损与水汽渗入：涉事飞机的右侧风挡玻璃边缘的封严硅胶出现了微小的破损。在长期的飞行中，外部环境的水汽通过这个破损处，渗入并存留在风挡玻璃底部的空腔内。\n2. 绝缘层性能下降：风挡玻璃内部装有用于防冰防雾的电加热系统。水汽的长期浸泡，导致加热系统底部导线的绝缘材料性能严重下降。\n3. 电弧放电（Arcing）：在事发当天，绝缘性能进一步恶化，导致导线之间或导线与周围结构之间产生了持续的电弧放电。\n4. 局部高温爆裂：电弧放电产生了极高温度的局部热点。风挡玻璃属于多层结构，受热极度不均，导致双层结构玻璃瞬间发生破裂。\n5. 无法承受压差脱落：玻璃结构被热应力破坏后，彻底丧失了结构强度，无法承受万米高空驾驶舱内外巨大的压力差，最终导致整块风挡玻璃从机身上爆裂脱落。\n\n调查报告明确指出，这是一次由极其隐蔽的机械缺陷引发的突发事故，在当时的常规航线维修检查中，是几乎无法通过目视检查提前发现的。\n\n第七部分：深远影响与历史意义\n\n1. 航空史上的奇迹：这次备降被世界航空界公认为“史诗级”的操作，难度甚至超越了著名的“哈德逊河奇迹”（全美航空1549号航班）。刘传健机长被授予“中国民航英雄机长”称号，3U8633机组被授予“中国民航英雄机组”称号。\n2. 促进行业安全升级：事故发生后，空客公司发布了针对A320系列飞机风挡玻璃的紧急服务通告，全球各大航空公司修改了维修手册，增加了对风挡玻璃加热系统绝缘电阻的定期测试，从源头上堵住了这一潜在的安全漏洞。\n3. 文化记忆：这一真实事件被改编为电影《中国机长》（2019年上映），让更多公众了解了民航人的不易与坚守，成为了中国民航安全文化的一座丰碑。"
    p0 = p0 * 20
    
    # 2. 获取基于对数正态分布的 Ratio 列表
    ratio = generate_ratio_list_lognormal_with_long(
        total_count,
        mu,
        sigma,
        num_long_prompts,
        seed=prompt_seed,
    )

    # 3. 排列并生成 Prompts (复用你现有的 gen_prompts 函数)
    prompts = gen_prompts(
        ratio,
        p0,
        output_mean_min=output_mean_min,
        output_mean_max=output_mean_max,
        output_curve_power=output_curve_power,
        output_std_dev=output_std_dev,
        output_min=output_min,
        output_max=output_max,
        output_seed=output_seed,
    )
    
    generation_metadata = {
        "total_count": total_count,
        "prompt_distribution": {
            "type": "lognormal_ratio",
            "formula": "R ~ LogNormal(mu, sigma); R = clip(R, 0, 1)",
            "mu": mu,
            "sigma": sigma,
            "num_long_prompts": num_long_prompts,
            "seed": prompt_seed,
        },
        "output_distribution": {
            "type": "conditional_truncated_normal",
            "formula": (
                "r_i=L_i/L_max; "
                "mu_i=mu_min+(mu_max-mu_min)*r_i^power; "
                "O_i~TruncNormal(mu_i, std_dev, min<=O_i<=max)"
            ),
            "mean_min": output_mean_min,
            "mean_max": output_mean_max,
            "curve_power": output_curve_power,
            "std_dev": output_std_dev,
            "min": output_min,
            "max": output_max,
            "seed": output_seed,
        },
    }
    return save_dataset_artifacts(
        file_path,
        prompts,
        generation_metadata,
        save_visualization=save_visualization,
        visualization_output_dir=visualization_output_dir,
        tokenizer_path=tokenizer_path,
    )


def generate_ratio_list(
    total_count,
    mean_percent,
    std_dev,
    num_long_prompts,
    seed=42,
):
    """
    根据给定的统计参数生成 ratio 列表。
    
    :param total_count: 总 prompt 个数
    :param mean_percent: 均值百分比 (例如输入 50 代表 50%, 即 0.5)
    :param std_dev: 标准差 (例如 0.15)
    :param num_long_prompts: 长度为 100% (ratio=1.0) 的长 prompt 个数
    :return: 包含所有 ratio 的 float 列表
    """
    if num_long_prompts > total_count:
        raise ValueError("❌ 长 prompt 的个数不能大于总 prompt 个数！")

    # 1. 将百分比转换为小数
    mean = mean_percent / 100.0

    # 2. 生成固定长度的长 prompt (100% -> 1.0)
    long_ratios = [1.0] * num_long_prompts

    # 3. 计算需要用正态分布生成的个数
    num_sampled = total_count - num_long_prompts
    
    sampled_ratios = []
    if num_sampled > 0:
        rng = np.random.default_rng(seed)
        # 使用 numpy 生成正态分布数据
        raw_ratios = rng.normal(loc=mean, scale=std_dev, size=num_sampled)
        
        # ⚠️ 关键操作：将生成的数据强制截断在 0.0 到 1.0 之间
        # (因为正态分布是有可能超出这个范围的)
        clipped_ratios = np.clip(raw_ratios, 0.0, 1.0)
        sampled_ratios = clipped_ratios.tolist()

    # 4. 合并列表并随机打乱
    final_ratio_list = long_ratios + sampled_ratios
    np.random.default_rng(seed + 1).shuffle(final_ratio_list)

    return final_ratio_list

def generate_ratio_list_lognormal_with_long(
    total_count,
    mu,
    sigma,
    num_long_prompts,
    seed=42,
):
    """
    基于对数正态分布生成 ratio 列表，并强制包含特定数量的长序列 (ratio=1.0)
    
    :param mu: 对数正态分布的均值 (注意：因为我们要生成 0~1 的比例，mu 通常是负数)
    :param sigma: 分布的标准差（决定长尾厚度）
    """
    if num_long_prompts > total_count:
        raise ValueError("❌ 长 prompt 的个数不能大于总 prompt 个数！")

    # 1. 生成固定数量的长 prompt (100% -> 1.0)
    long_ratios = [1.0] * num_long_prompts

    # 2. 计算剩余需要采样的数量
    num_sampled = total_count - num_long_prompts

    sampled_ratios = []
    if num_sampled > 0:
        rng = np.random.default_rng(seed)
        # 使用对数正态分布生成
        raw_ratios = rng.lognormal(
            mean=mu,
            sigma=sigma,
            size=num_sampled,
        )
        
        # 强制截断在 0.0 到 1.0 之间
        clipped_ratios = np.clip(raw_ratios, 0.0, 1.0)
        sampled_ratios = clipped_ratios.tolist()

    # 3. 合并列表并随机打乱
    final_ratio_list = long_ratios + sampled_ratios
    np.random.default_rng(seed + 1).shuffle(final_ratio_list)

    return final_ratio_list


def generate_output_token_lengths(
    prompt_ratios,
    mean_min=96,
    mean_max=384,
    curve_power=0.7,
    std_dev=48,
    min_value=64,
    max_value=512,
    seed=42,
):
    """按 prompt 长度生成条件截断正态分布的输出长度。"""
    prompt_ratios = np.asarray(prompt_ratios, dtype=np.float64)
    if prompt_ratios.ndim != 1:
        raise ValueError("prompt_ratios 必须是一维序列")
    if np.any((prompt_ratios < 0) | (prompt_ratios > 1)):
        raise ValueError("prompt_ratios 必须位于 [0, 1] 范围内")
    if curve_power <= 0:
        raise ValueError("curve_power 必须大于 0")
    if std_dev <= 0:
        raise ValueError("std_dev 必须大于 0")
    if min_value < 1:
        raise ValueError("min_value 必须至少为 1")
    if min_value >= max_value:
        raise ValueError("min_value 必须小于 max_value")
    if not min_value <= mean_min <= mean_max <= max_value:
        raise ValueError(
            "必须满足 min_value <= mean_min <= mean_max <= max_value"
        )

    rng = np.random.default_rng(seed)
    conditional_means = mean_min + (mean_max - mean_min) * np.power(
        prompt_ratios,
        curve_power,
    )
    sampled_values = []

    # 每条请求都有自己的 mu_i，拒绝采样避免结果堆积在截断边界。
    for conditional_mean in conditional_means:
        while True:
            candidate = rng.normal(loc=conditional_mean, scale=std_dev)
            if min_value <= candidate <= max_value:
                sampled_values.append(int(round(candidate)))
                break

    return sampled_values


def gen_prompts(
    ratio_list,
    p0,
    output_mean_min=96,
    output_mean_max=384,
    output_curve_power=0.7,
    output_std_dev=48,
    output_min=64,
    output_max=512,
    output_seed=42,
):
    # 1. 性能优化：在循环外只切分一次长 prompt
    words = p0.split()
    total_len = len(words)
    if total_len == 0:
        raise ValueError("p0 不能为空")

    safe_ratios = np.clip(
        np.asarray(ratio_list, dtype=np.float64),
        0.0,
        1.0,
    )
    target_word_counts = (total_len * safe_ratios).astype(np.int64)
    normalized_prompt_lengths = target_word_counts / total_len
    
    result_prompts = []
    
    output_token_lengths = generate_output_token_lengths(
        prompt_ratios=normalized_prompt_lengths,
        mean_min=output_mean_min,
        mean_max=output_mean_max,
        curve_power=output_curve_power,
        std_dev=output_std_dev,
        min_value=output_min,
        max_value=output_max,
        seed=output_seed,
    )

    # 2. 遍历每条请求的实际目标词数
    for target_word_count, output_tokens in zip(
        target_word_counts,
        output_token_lengths,
    ):
        # 截取并拼接
        truncated_words = words[:target_word_count]
        full_prompt_string = " ".join(truncated_words)
        result_prompts.append(
            {
                "prompt": full_prompt_string,
                "output_tokens": output_tokens,
            }
        )
        # print(f"prompt: {truncated_words}")

        # result_prompts.append({"prompt": "写篇1w字的作文，题目是我的区长父亲。", "output_len": 20}) # 测试用
        # result_prompts.append({"prompt": p0, "output_len": 20}) # 测试用
        
    return result_prompts

def main():
    args = parse_args()
    dataset_path = args.dataset_path or default_dataset_path()

    # ==========================
    # 正态分布策略
    # ==========================
    # tot_num, mean, std, num_long = 1000, 10, 0.05, 20
    # generate_mixed_dataset("mixed_prompts_normal.jsonl", tot_num, mean, std, num_long)
    # print(f"✅ 数据集生成完毕：mixed_prompts_normal.jsonl | tot_num={tot_num}, mean={mean}, std={std}, num_long={num_long}")

    # ==========================
    # 对数正态策略
    # ==========================
    # mu = -2.3 (代表峰值大约在 10% 的比例处)
    # sigma = 0.8 (决定分布有多宽)
    tot_num, mu, sigma, num_long = 1000, -2.5, 0.7, 20
    output_mean_min, output_mean_max = 96, 384
    output_curve_power, output_std_dev = 0.7, 48
    output_min, output_max, output_seed = 64, 512, 42
    prompt_seed = 42
    tokenizer_path = None  # 建议在服务器上填写实际模型或 tokenizer 路径
    
    artifacts = generate_mixed_dataset_lognormal(
        dataset_path,
        tot_num,
        mu,
        sigma,
        num_long,
        output_mean_min=output_mean_min,
        output_mean_max=output_mean_max,
        output_curve_power=output_curve_power,
        output_std_dev=output_std_dev,
        output_min=output_min,
        output_max=output_max,
        output_seed=output_seed,
        prompt_seed=prompt_seed,
        save_visualization=True,
        visualization_output_dir="visualization",
        tokenizer_path=tokenizer_path,
    )
    print(
        f"✅ 对数正态数据集生成完毕：{dataset_path}"
        f" | tot_num={tot_num}, mu={mu}, sigma={sigma}, num_long={num_long}"
        " | output_mu="
        f"{output_mean_min}+({output_mean_max}-{output_mean_min})"
        f"*(prompt_len/max_prompt_len)^{output_curve_power}"
        f" | output_tokens~TruncNormal(mu, std={output_std_dev}, "
        f"min={output_min}, max={output_max}, seed={output_seed})"
    )
    for artifact_name, artifact_path in artifacts.items():
        print(f"  {artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()
