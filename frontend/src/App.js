import React, { useState, useEffect } from 'react';
import { Layout, Menu, DatePicker, Card, Row, Col, Statistic, Spin, Alert } from 'antd';
import { LineChartOutlined, BarChartOutlined, DashboardOutlined } from '@ant-design/icons';
import axios from 'axios';
import moment from 'moment';
import './App.css';

// Import chart components
import PriceChart from './components/PriceChart';
import VolatilityChart from './components/VolatilityChart';
import EventsTimeline from './components/EventsTimeline';
import AnalyticsPanel from './components/AnalyticsPanel';

const { Header, Content, Sider } = Layout;
const { RangePicker } = DatePicker;

function App() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [changePoints, setChangePoints] = useState([]);
  const [events, setEvents] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [dateRange, setDateRange] = useState(null);
  const [selectedView, setSelectedView] = useState('overview');
  const [error, setError] = useState(null);

  // Load initial data
  useEffect(() => {
    loadData();
    loadChangePoints();
    loadEvents();
    loadAnalytics();
  }, []);

  const loadData = async (filters = null) => {
    try {
      setLoading(true);
      let response;
      
      if (filters) {
        response = await axios.post('/api/data/filtered', filters);
      } else {
        response = await axios.get('/api/data');
      }
      
      setData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load price data');
      console.error('Error loading data:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadChangePoints = async () => {
    try {
      const response = await axios.get('/api/changepoints');
      setChangePoints(response.data.change_points || []);
    } catch (err) {
      console.error('Error loading change points:', err);
    }
  };

  const loadEvents = async () => {
    try {
      const response = await axios.get('/api/events');
      setEvents(response.data.events || []);
    } catch (err) {
      console.error('Error loading events:', err);
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await axios.get('/api/analytics');
      setAnalytics(response.data);
    } catch (err) {
      console.error('Error loading analytics:', err);
    }
  };

  const handleDateRangeChange = (dates) => {
    if (dates && dates.length === 2) {
      const filters = {
        start_date: dates[0].format('YYYY-MM-DD'),
        end_date: dates[1].format('YYYY-MM-DD')
      };
      setDateRange(dates);
      loadData(filters);
    } else {
      setDateRange(null);
      loadData();
    }
  };

  const menuItems = [
    {
      key: 'overview',
      icon: <DashboardOutlined />,
      label: 'Overview',
    },
    {
      key: 'price-analysis',
      icon: <LineChartOutlined />,
      label: 'Price Analysis',
    },
    {
      key: 'volatility',
      icon: <BarChartOutlined />,
      label: 'Volatility Analysis',
    }
  ];

  const renderContent = () => {
    if (loading) {
      return (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
          <p>Loading dashboard data...</p>
        </div>
      );
    }

    if (error) {
      return (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          style={{ margin: '20px' }}
        />
      );
    }

    if (!data) {
      return (
        <Alert
          message="No Data"
          description="No data available to display"
          type="warning"
          showIcon
          style={{ margin: '20px' }}
        />
      );
    }

    switch (selectedView) {
      case 'overview':
        return (
          <div>
            {/* Statistics Cards */}
            <Row gutter={16} style={{ marginBottom: '20px' }}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Current Price"
                    value={data.prices[data.prices.length - 1]}
                    precision={2}
                    prefix="$"
                    suffix="/barrel"
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Average Price"
                    value={data.statistics.avg_price}
                    precision={2}
                    prefix="$"
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Price Range"
                    value={`$${data.statistics.min_price.toFixed(2)} - $${data.statistics.max_price.toFixed(2)}`}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Total Records"
                    value={data.statistics.total_records}
                  />
                </Card>
              </Col>
            </Row>

            {/* Main Chart */}
            <Card title="Brent Oil Price Trends with Change Points" style={{ marginBottom: '20px' }}>
              <PriceChart 
                data={data} 
                changePoints={changePoints}
                events={events}
              />
            </Card>

            {/* Events Timeline */}
            <Card title="Historical Events Impact">
              <EventsTimeline events={events} />
            </Card>
          </div>
        );

      case 'price-analysis':
        return (
          <div>
            <Card title="Detailed Price Analysis" style={{ marginBottom: '20px' }}>
              <PriceChart 
                data={data} 
                changePoints={changePoints}
                events={events}
                detailed={true}
              />
            </Card>
            
            {analytics && (
              <AnalyticsPanel analytics={analytics} />
            )}
          </div>
        );

      case 'volatility':
        return (
          <Card title="Volatility Analysis">
            {analytics && (
              <VolatilityChart 
                data={data}
                volatility={analytics.volatility}
                returns={analytics.returns}
              />
            )}
          </Card>
        );

      default:
        return <div>Select a view from the menu</div>;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', padding: '0 20px' }}>
        <div style={{ color: 'white', fontSize: '18px', fontWeight: 'bold' }}>
          Brent Oil Price Analysis Dashboard - Birhan Energies
        </div>
      </Header>
      
      <Layout>
        <Sider width={250} style={{ background: '#fff' }}>
          <div style={{ padding: '20px' }}>
            <h4>Date Range Filter</h4>
            <RangePicker
              value={dateRange}
              onChange={handleDateRangeChange}
              style={{ width: '100%', marginBottom: '20px' }}
            />
          </div>
          
          <Menu
            mode="inline"
            selectedKeys={[selectedView]}
            items={menuItems}
            onClick={({ key }) => setSelectedView(key)}
          />
        </Sider>
        
        <Layout style={{ padding: '20px' }}>
          <Content style={{ background: '#fff', padding: '20px', minHeight: 280 }}>
            {renderContent()}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
}

export default App;
