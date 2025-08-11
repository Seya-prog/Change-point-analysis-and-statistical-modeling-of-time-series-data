import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import moment from 'moment';

const EventsTimeline = ({ events = [] }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!events.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 800 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Parse dates and sort events
    const sortedEvents = events
      .map(event => ({
        ...event,
        parsedDate: moment(event.date).toDate()
      }))
      .sort((a, b) => a.parsedDate - b.parsedDate);

    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(sortedEvents, d => d.parsedDate))
      .range([0, width]);

    const colorScale = d3.scaleOrdinal()
      .domain(['conflict', 'economic', 'policy', 'terrorism', 'pandemic'])
      .range(['#ff4d4f', '#1890ff', '#52c41a', '#faad14', '#722ed1']);

    // Create timeline line
    g.append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", height / 2)
      .attr("y2", height / 2)
      .attr("stroke", "#d9d9d9")
      .attr("stroke-width", 2);

    // Add x-axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.timeFormat("%Y"));

    g.append("g")
      .attr("transform", `translate(0,${height - 20})`)
      .call(xAxis);

    // Add events
    const eventGroups = g.selectAll(".event")
      .data(sortedEvents)
      .enter()
      .append("g")
      .attr("class", "event")
      .attr("transform", (d, i) => `translate(${xScale(d.parsedDate)},${height / 2})`);

    // Add event circles
    eventGroups.append("circle")
      .attr("r", 6)
      .attr("fill", d => colorScale(d.type))
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .style("cursor", "pointer");

    // Add event labels
    eventGroups.append("text")
      .attr("dy", -15)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .style("font-weight", "bold")
      .text(d => d.title.length > 15 ? d.title.substring(0, 15) + "..." : d.title);

    // Add tooltips
    const tooltip = d3.select("body").append("div")
      .attr("class", "d3-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "10px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("max-width", "200px");

    eventGroups
      .on("mouseover", function(event, d) {
        tooltip.style("visibility", "visible")
          .html(`
            <strong>${d.title}</strong><br/>
            <em>${moment(d.date).format('MMMM DD, YYYY')}</em><br/>
            ${d.description}<br/>
            <span style="color: ${colorScale(d.type)}">Impact: ${d.impact}</span>
          `);
      })
      .on("mousemove", function(event) {
        tooltip.style("top", (event.pageY - 10) + "px")
          .style("left", (event.pageX + 10) + "px");
      })
      .on("mouseout", function() {
        tooltip.style("visibility", "hidden");
      });

    // Cleanup function
    return () => {
      d3.select("body").selectAll(".d3-tooltip").remove();
    };

  }, [events]);

  return (
    <div>
      <div style={{ marginBottom: '10px' }}>
        <span style={{ fontSize: '14px', color: '#666' }}>
          Historical Events Timeline (hover for details)
        </span>
      </div>
      <svg ref={svgRef}></svg>
      <div style={{ marginTop: '10px', display: 'flex', gap: '15px', fontSize: '12px' }}>
        <span><span style={{ color: '#ff4d4f' }}>●</span> Conflict</span>
        <span><span style={{ color: '#1890ff' }}>●</span> Economic</span>
        <span><span style={{ color: '#52c41a' }}>●</span> Policy</span>
        <span><span style={{ color: '#faad14' }}>●</span> Terrorism</span>
        <span><span style={{ color: '#722ed1' }}>●</span> Pandemic</span>
      </div>
    </div>
  );
};

export default EventsTimeline;
