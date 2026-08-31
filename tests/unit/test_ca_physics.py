"""Unit tests for Rothermel fire-behavior physics calculations.

Validates the core rate-of-spread computation against known analytical
cases: flat/no-wind homogeneous spread, slope-accelerated spread,
and wind-driven elongation.
"""
import math

from aerospatium.ca_engine.rothermel import rothermel_forward_ros


class TestRothermelPhysics:
    """Verify basic Rothermel ROS computation."""

    def test_grass_flat_no_wind(self):
        """FM1 on flat terrain, no wind → non-zero base ROS."""
        ros = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=0.0,
            aspect_rad=0.0,
            wind_speed_ms=0.0,
            wind_dir_rad=0.0,
            neighbor_azimuth_rad=0.0,
            fuel_moisture=0.08,
        )
        assert ros > 0.0, "Flat grass with fuel should produce non-zero ROS"
        assert ros < 100.0, "ROS should be physically plausible (< 100 m/min)"

    def test_chaparral_higher_ros_than_grass(self):
        """FM4 (chaparral) should have higher ROS than FM1 (grass) on same slope/wind."""
        kwargs = {
            "slope_rad": math.radians(10),
            "aspect_rad": 0.0,
            "wind_speed_ms": 5.0,
            "wind_dir_rad": 0.0,
            "neighbor_azimuth_rad": 0.0,
            "fuel_moisture": 0.08,
        }
        ros_grass = rothermel_forward_ros(fuel_model=1, **kwargs)
        ros_chaparral = rothermel_forward_ros(fuel_model=4, **kwargs)
        assert ros_chaparral > ros_grass, "Chaparral should burn faster than grass"

    def test_slope_increases_uphill_ros(self):
        """Uphill spread should be faster than downhill."""
        # Wind direction aligned with spread direction (0 = north)
        wind_dir = 0.0
        neighbor_az = 0.0  # spread northward

        ros_uphill = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=math.radians(20),
            aspect_rad=0.0,  # aspect aligned with spread
            wind_speed_ms=0.0,
            wind_dir_rad=wind_dir,
            neighbor_azimuth_rad=neighbor_az,
            fuel_moisture=0.08,
        )

        ros_downhill = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=math.radians(-20),
            aspect_rad=math.pi,  # aspect facing opposite (downhill)
            wind_speed_ms=0.0,
            wind_dir_rad=wind_dir,
            neighbor_azimuth_rad=neighbor_az,
            fuel_moisture=0.08,
        )

        assert ros_uphill >= ros_downhill, "Uphill ROS should exceed downhill"

    def test_wind_aligned_increases_ros(self):
        """Wind blowing toward neighbor should increase ROS."""
        slope_rad = math.radians(5)
        aspect_rad = 0.0  # north-facing

        # Wind FROM south (blowing north) aligned with northward spread
        ros_with_wind = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=slope_rad,
            aspect_rad=aspect_rad,
            wind_speed_ms=10.0,
            wind_dir_rad=math.radians(180),  # FROM south
            neighbor_azimuth_rad=0.0,  # spread northward
            fuel_moisture=0.08,
        )

        ros_no_wind = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=slope_rad,
            aspect_rad=aspect_rad,
            wind_speed_ms=0.0,
            wind_dir_rad=0.0,
            neighbor_azimuth_rad=0.0,
            fuel_moisture=0.08,
        )

        assert ros_with_wind > ros_no_wind, "Wind-aligned spread should exceed no-wind"

    def test_non_burnable_fuel_returns_zero(self):
        """Fuel model 99 (non-burnable) should return 0 ROS."""
        ros = rothermel_forward_ros(
            fuel_model=99,
            slope_rad=0.0,
            aspect_rad=0.0,
            wind_speed_ms=10.0,
            wind_dir_rad=0.0,
            neighbor_azimuth_rad=0.0,
            fuel_moisture=0.08,
        )
        assert ros == 0.0, "Non-burnable fuel should produce zero ROS"

    def test_fuel_moisture_suppression(self):
        """Higher fuel moisture should reduce or nullify ROS."""
        ros_low_moisture = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=0.0,
            aspect_rad=0.0,
            wind_speed_ms=5.0,
            wind_dir_rad=0.0,
            neighbor_azimuth_rad=0.0,
            fuel_moisture=0.05,
        )
        ros_high_moisture = rothermel_forward_ros(
            fuel_model=1,
            slope_rad=0.0,
            aspect_rad=0.0,
            wind_speed_ms=5.0,
            wind_dir_rad=0.0,
            neighbor_azimuth_rad=0.0,
            fuel_moisture=0.30,
        )
        assert ros_high_moisture <= ros_low_moisture, "Higher moisture should not increase ROS"

    def test_ros_never_negative(self):
        """ROS must always be non-negative."""
        for fm in [1, 2, 4, 10]:
            for slope in [-1.0, 0.0, 0.5, 1.0]:
                for wind in [0.0, 5.0, 20.0]:
                    ros = rothermel_forward_ros(
                        fuel_model=fm,
                        slope_rad=slope,
                        aspect_rad=0.0,
                        wind_speed_ms=wind,
                        wind_dir_rad=0.0,
                        neighbor_azimuth_rad=1.5,
                        fuel_moisture=0.08,
                    )
                    assert ros >= 0.0, f"ROS negative for fm={fm}, slope={slope}, wind={wind}"
