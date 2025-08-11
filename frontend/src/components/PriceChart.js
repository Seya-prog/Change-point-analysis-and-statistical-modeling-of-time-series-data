import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Scatter,
  ComposedChart
} from 'recharts';
import moment from 'moment';

const PriceChart = ({ data, changePoints = [], events = [], detailed = false }) => {
  if (!data || !data.dates || !data.prices) {
    return <div>No data available</div>;
  }

  // Prepare chart data
  const chartData = data.dates.map((date, index) => ({
    date: date,
    price: data.prices[index],
    formattedDate: moment(date).format('MMM YYYY')
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const date = moment(label).format('MMMM DD, YYYY');
      const price = payload[0].value;
      
      // Check if this date has an event
      const event = events.find(e => moment(e.date).isSame(moment(label), 'day'));
      
      return (
        <div style={{
          backgroundColor: 'white',
          padding: '10px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <p style={{ margin: 0, fontWeight: 'bold' }}>{date}</p>
          <p style={{ margin: 0, color: '#1890ff' }}>
            Price: ${price?.toFixed(2)}/barrel
          </p>
          {event && (
            <div style={{ marginTop: '8px', padding: '4px', backgroundColor: '#fff2e8', borderRadius: '2px' }}>
              <p style={{ margin: 0, fontWeight: 'bold', color: '#d46b08' }}>
                {event.title}
              </p>
              <p style={{ margin: 0, fontSize: '12px', color: '#8c8c8c' }}>
                {event.description}
              </p>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height={detailed ? 500 : 400}>
      <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="date"
          tickFormatter={(value) => moment(value).format('MMM YY')}
          interval="preserveStartEnd"
        />
        <YAxis 
          label={{ value: 'Price (USD/barrel)', angle: -90, position: 'insideLeft' }}
          domain={['dataMin - 5', 'dataMax + 5']}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        
        {/* Main price line */}
        <Line
          type="monotone"
          dataKey="price"
          stroke="#1890ff"
          strokeWidth={2}
          dot={false}
          name="Brent Oil Price"
        />
        
        {/* Change points as vertical lines */}
        {changePoints.map((cp, index) => (
          <ReferenceLine
            key={`cp-${index}`}
            x={cp}
            stroke="#ff4d4f"
            strokeDasharray="5 5"
            strokeWidth={2}
            label={{ value: "Change Point", position: "top" }}
          />
        ))}
        
        {/* Major events as vertical lines */}
        {events.map((event, index) => (
          <ReferenceLine
            key={`event-${index}`}
            x={event.date}
            stroke="#faad14"
            strokeDasharray="3 3"
            strokeWidth={1}
            label={{ 
              value: event.title.substring(0, 15) + "...", 
              position: "topLeft",
              fontSize: 10
            }}
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;
