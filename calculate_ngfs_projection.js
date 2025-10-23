#!/usr/bin/env node

/**
 * Self-contained Node.js translation of calculate_ngfs_projection from util.py
 *
 * This script calculates NGFS (Network for Greening the Financial System) projections
 * for energy production or emissions based on country-specific data and scenarios.
 *
 * Usage:
 *   npm install data-forge data-forge-fs
 *   node calculate_ngfs_projection.js
 */

const dataForge = require('data-forge');
const fs = require('fs');
const zlib = require('zlib');

// Constants from util.py
const SUBSECTORS = ['Coal', 'Oil', 'Gas'];
const hours_in_1year = 24 * 365.25;
const seconds_in_1hour = 3600;
// Match Python's CARBON_BUDGET_CONSISTENT setting
const CARBON_BUDGET_CONSISTENT = '15-50';
const ENABLE_AI = false;

// Helper function: Convert coal to GJ
function coal2GJ(x) {
  const mul = 29.3076; // 1 tce is 29.3076 GJ
  return Array.isArray(x) ? x.map(e => e * mul) : x * mul;
}

// Helper function: Convert GJ to coal
function GJ2coal(x) {
  return x / 29.3076;
}

// Helper function: Convert GJ to MWh
function GJ2MWh(x) {
  return (x * 1e9 / 3600) / 1e6;
}

// Helper function: Convert coal to MWh
function coal2MWh(x) {
  return GJ2MWh(coal2GJ(x));
}

/**
 * Read NGFS data files
 * @returns {Object} Object containing production and emissions dataframes
 */
function readNGFS() {
  console.log('Reading NGFS data...');

  // Decompress gzipped files first, then parse CSV
  const productionGz = fs.readFileSync('data/2-GCAM6-filtered-prim-and-secon-energy.csv.gz');
  const productionCsv = zlib.gunzipSync(productionGz).toString();
  const production = dataForge.fromCSV(productionCsv);

  const emissionsGz = fs.readFileSync('data/3-GCAM6-emissions.csv.gz');
  const emissionsCsv = zlib.gunzipSync(emissionsGz).toString();
  const emissions = dataForge.fromCSV(emissionsCsv);

  console.log(`Loaded ${production.count()} production rows`);
  console.log(`Loaded ${emissions.count()} emissions rows`);

  return { production, emissions };
}

/**
 * Read carbon budget consistent data for Net Zero 2050 scenario
 * @param {string} carbonBudgetConsistent - Version identifier ('15-50', '15-67', '16-67', 'strictly_declining')
 * @returns {DataFrame} Carbon budget consistent dataframe
 */
function readCarbonBudgetConsistent(carbonBudgetConsistent) {
  const fileMap = {
    '15-50': '6.1-NZ-15-50-v2-Secondary-annual.csv',
    '15-67': '7.1-NZ-15-67-v2-Secondary-annual.csv',
    '16-67': '8.1-NZ-16-67-v2-Secondary-annual.csv',
    'strictly_declining': '10.1 - NZ-15-50 - v3 - Secondary - annual_emissions.csv'
  };

  const filename = fileMap[carbonBudgetConsistent];
  if (!filename) {
    throw new Error(`Unknown carbon budget consistent mode: ${carbonBudgetConsistent}`);
  }

  console.log(`Reading carbon budget consistent data: ${filename}`);
  const csvData = fs.readFileSync(`./data_private/${filename}`, 'utf-8');
  return dataForge.fromCSV(csvData);
}

/**
 * Get time series data for a specific variable from NGFS data
 * @param {DataFrame} ngfsData - NGFS dataframe
 * @param {string} variableName - Variable name to filter
 * @param {Array<number>} years - Years to extract
 * @returns {Array<number>|null} Time series data or null if not found
 */
function getNGFSTimeseriesForVariable(ngfsData, variableName, years) {
  const filtered = ngfsData.where(row => row.Variable === variableName);

  if (filtered.count() === 0) {
    return null;
  }

  const row = filtered.first();
  return years.map(year => parseFloat(row[year.toString()]) || 0);
}

/**
 * Main function: Calculate NGFS projection
 *
 * @param {string} productionOrEmissions - 'production' or 'emissions'
 * @param {Object} valueFa - Forward Analytics value by country and subsector
 *                          Format: { 'US': { 'Coal': 123, 'Gas': 456 }, ... }
 * @param {Object} ngfsData - NGFS dataframes object with 'production' and 'emissions' keys
 * @param {string} sector - Sector name (e.g., 'Power')
 * @param {string} scenario - NGFS scenario name (e.g., 'Net Zero 2050', 'Current Policies')
 * @param {number} startYear - Start year for projection
 * @param {number} lastYear - End year for projection
 * @param {Object} alpha2ToAlpha3 - Mapping from ISO Alpha-2 to Alpha-3 country codes
 * @param {string|null} filterSubsector - Optional: filter to specific subsector
 * @param {DataFrame|null} unitProfitDf - Optional: unit profit data for profit calculation
 * @returns {Object} { timeseries, summed, profit }
 */
function calculateNGFSProjection(
  productionOrEmissions,
  valueFa,
  ngfsData,
  sector,
  scenario,
  startYear,
  lastYear,
  alpha2ToAlpha3,
  filterSubsector = null,
  unitProfitDf = null
) {
  console.log(`\nCalculating NGFS projection:`);
  console.log(`  Mode: ${productionOrEmissions}`);
  console.log(`  Sector: ${sector}`);
  console.log(`  Scenario: ${scenario}`);
  console.log(`  Years: ${startYear}-${lastYear}`);

  if (sector !== 'Power') {
    throw new Error('Only Power sector is currently supported');
  }

  let ngfs = ngfsData[productionOrEmissions];
  ngfs = ngfs.where(row => row.Scenario === scenario);

  // Handle special scenarios like Python version
  if (scenario === 'Current Policies' && ENABLE_AI) {
    ngfs = ngfsData['emissions_ai'];
    ngfs = ngfs.where(row => row.Scenario === scenario);
  } else if (scenario === 'Net Zero 2050' && CARBON_BUDGET_CONSISTENT) {
    ngfs = readCarbonBudgetConsistent(CARBON_BUDGET_CONSISTENT);
  }

  const yearsInterpolated = Array.from(
    { length: lastYear - startYear + 1 },
    (_, i) => startYear + i
  );

  const subsectors = filterSubsector ? [filterSubsector] : SUBSECTORS;
  const countries = Object.keys(valueFa);
  const out = {};
  const outProfit = {};

  // Initialize profit structure
  subsectors.forEach(subsector => {
    outProfit[subsector] = {};
  });

  // Find "Countries without IEA statistics" data for fallback
  const ngfsCountryWoIeaStats = ngfs.where(
    row => row.Region === 'Downscaling|Countries without IEA statistics'
  );

  /**
   * Country code mapper (handles special cases)
   */
  function countryMapper(alpha2) {
    if (alpha2 === 'XK') { // Kosovo
      return 'XKX';
    }
    return alpha2ToAlpha3[alpha2] || alpha2;
  }

  // Process each country
  countries.forEach(country => {
    const alpha3 = countryMapper(country);
    let ngfsCountry = ngfs.where(row => row.Region === alpha3);

    let unitProfitCountry = null;
    if (unitProfitDf) {
      const profitRow = unitProfitDf.where(row => row['Alpha-2 Code'] === country);
      unitProfitCountry = profitRow.count() > 0 ? profitRow.first() : 0;
    }

    // Use fallback if no country-specific data
    const ngfsSource = ngfsCountry.count() > 0 ? ngfsCountry : ngfsCountryWoIeaStats;
    const valueFaCountry = valueFa[country];

    subsectors.forEach(subsector => {
      if (!valueFaCountry || !(subsector in valueFaCountry)) {
        return;
      }

      const variable = `Secondary Energy|Electricity|${subsector}`;

      // Try to get country-specific NGFS data
      let acrossYears = getNGFSTimeseriesForVariable(
        ngfsSource,
        variable,
        yearsInterpolated
      );

      // Fallback to global stats if no data or starts at zero
      if (!acrossYears || acrossYears[0] === 0) {
        acrossYears = getNGFSTimeseriesForVariable(
          ngfsCountryWoIeaStats,
          variable,
          yearsInterpolated
        );
      }

      if (!acrossYears) {
        console.warn(`No data found for ${country} (${alpha3}) - ${subsector}`);
        return;
      }

      // Rescale NGFS projection to match FA value at start year
      if (acrossYears[0] !== 0) {
        const scalingFactor = valueFaCountry[subsector] / acrossYears[0];
        acrossYears = acrossYears.map(val => val * scalingFactor);
      } else {
        acrossYears = new Array(acrossYears.length).fill(0);
      }

      // Store production timeseries
      const key = `${country}__${subsector}`;
      out[key] = acrossYears;

      // Calculate profit if unit profit data is provided
      if (unitProfitDf && typeof unitProfitCountry !== 'number') {
        const unitProfitValue = unitProfitCountry[`${subsector}_Av_Profitability_$/MWh`] || 0;

        // Cap unit profit to never be negative
        const cappedUnitProfit = Math.max(unitProfitValue, 0);

        // Convert to profit (multiply by 1e9 for Giga tonnes conversion)
        const acrossYearsProfit = acrossYears.map(
          e => coal2MWh(e) * 1e9 * cappedUnitProfit
        );

        outProfit[subsector][country] = acrossYearsProfit;
      }
    });
  });

  // Calculate total sum
  const summed = Object.values(out).reduce((sum, values) =>
    sum + values.reduce((a, b) => a + b, 0), 0
  );

  // Restructure output to match Python format
  const finalOut = yearsInterpolated.map((_, i) => {
    const yearData = {};
    Object.entries(out).forEach(([key, values]) => {
      yearData[key] = values[i];
    });
    return yearData;
  });

  // Restructure profit output
  const finalOutProfit = {};
  if (unitProfitDf) {
    subsectors.forEach(subsector => {
      finalOutProfit[subsector] = yearsInterpolated.map((_, i) => {
        const yearData = {};
        Object.entries(outProfit[subsector]).forEach(([country, values]) => {
          yearData[country] = values[i];
        });
        return yearData;
      });
    });
  }

  // For production mode, sum across subsectors
  if (productionOrEmissions === 'production') {
    finalOut.forEach(yearData => {
      const summedByCountry = {};
      Object.entries(yearData).forEach(([key, value]) => {
        const country = key.split('__')[0];
        summedByCountry[country] = (summedByCountry[country] || 0) + value;
      });
      Object.assign(yearData, summedByCountry);
    });
  }

  console.log(`  Total summed: ${summed.toFixed(2)}`);
  console.log(`  Timeseries length: ${finalOut.length}`);

  return {
    timeseries: finalOut,
    summed,
    profit: finalOutProfit
  };
}

/**
 * Example usage
 */
function main() {
  try {
    // Read NGFS data
    const ngfsData = readNGFS();

    // Example alpha2 to alpha3 mapping (partial - add more as needed)
    const alpha2ToAlpha3 = {
      'IN': 'IND',
      'XK': 'XKX',  // Kosovo
      // Add more mappings as needed
    };

    // Example value_fa data (you would provide this)
    // Format: { countryCode: { subsector: value } }
    const valueFa = {
      'IN': { 'Coal': 0.142901, 'Gas': 0.016550, 'Oil': 0.000395, 'Other': 0.000257 }
    };

    // Calculate projection
    const result = calculateNGFSProjection(
      'production',        // production or emissions
      valueFa,             // Your FA data
      ngfsData,            // NGFS data
      'Power',             // Sector
      'Net Zero 2050',     // Scenario
      //'Current Policies',     // Scenario
      2024,                // Start year
      2050,                // End year
      alpha2ToAlpha3,      // Country code mapping
      null,                // Filter subsector (null = all)
      null                 // Unit profit data (null = no profit calc)
    );

    // Output results
    console.log('\n=== Results ===');
    console.log(`Total summed value: ${result.summed}`);
    console.log(`\nFirst year projection (2024):`);
    console.log(JSON.stringify(result.timeseries[0], null, 2));
    console.log(`\nLast year projection (2050):`);
    console.log(JSON.stringify(result.timeseries[result.timeseries.length - 1], null, 2));
    console.log(JSON.stringify(result.timeseries.map(e => e["IN"]), null, 2))

    // Save to file
    const outputFile = 'ngfs_projection_output.json';
    fs.writeFileSync(outputFile, JSON.stringify(result, null, 2));
    console.log(`\nResults saved to ${outputFile}`);

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
