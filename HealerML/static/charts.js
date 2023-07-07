// < !--饼状图 begin-- >
// 初始化 ECharts 实例
var chart = echarts.init(document.getElementById('pie chart'));

// 配置图表选项
var option = {
    title: {
        text: '各类别所占比例',
        subtext: '数据来源',
        left: 'center'
    },
    tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b} : {c} ({d}%)'
    },
    legend: {
        orient: 'vertical',
        left: 'left',
        data: ['类别1', '类别2', '类别3', '类别4', '类别5']
    },
    series: [
        {
            name: '数据统计',
            type: 'pie',
            radius: '55%',
            center: ['50%', '60%'],
            data: [
                { value: 335, name: '类别1' },
                { value: 310, name: '类别2' },
                { value: 234, name: '类别3' },
                { value: 135, name: '类别4' },
                { value: 1548, name: '类别5' }
            ],
            itemStyle: {
                emphasis: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            }
        }
    ]
};

// 使用配置项显示图表
chart.setOption(option);
// < !--饼状图 end-- >

// 折线图 begin
// 初始化 ECharts 实例
var chart = echarts.init(document.getElementById('line chart'));

// 配置图表选项
var option = {
    title: {
        text: '训练集 | 验证集Loss',
        left: 'center',
    },
    tooltip: {
        trigger: 'axis'
    },
    legend: {
        data: ['训练集Loss', '验证集Loss'],
        left: '10%',
        top: '7%'
    },
    xAxis: {
        type: 'category',
        data: ['A', 'B', 'C', 'D', 'E'],
        axisLabel: {
            interval: 0  // 强制显示所有标签
        }
    },
    yAxis: {
        type: 'value'
    },
    series: [
        {
            name: '训练集Loss',
            type: 'line',
            data: [10, 20, 30, 40, 50]
        },
        {
            name: '验证集Loss',
            type: 'line',
            data: [20, 40, 10, 50, 30]
        }
    ]
};

// 使用配置项显示图表
chart.setOption(option);
// 折线图 end