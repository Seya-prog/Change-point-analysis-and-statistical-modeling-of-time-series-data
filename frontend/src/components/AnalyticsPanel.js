import React from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';

const AnalyticsPanel = ({ analytics }) => {
  if (!analytics || !analytics.metrics) {
    return <div>No analytics data available</div>;
  }

  const { metrics } = analytics;

  return (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <Statistic
            title="Average Volatility"
            value={metrics.avg_volatility}
            precision={2}
            valueStyle={{ color: '#cf1322' }}
            prefix="$"
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="Maximum Volatility"
            value={metrics.max_volatility}
            precision={2}
            valueStyle={{ color: '#cf1322' }}
            prefix="$"
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="Average Daily Return"
            value={metrics.avg_return}
            precision={2}
            valueStyle={{ color: metrics.avg_return >= 0 ? '#3f8600' : '#cf1322' }}
            prefix={metrics.avg_return >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
            suffix="%"
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="Return Range"
            value={`${metrics.min_return.toFixed(2)}% to ${metrics.max_return.toFixed(2)}%`}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
    </Row>
  );
};

export default AnalyticsPanel;
