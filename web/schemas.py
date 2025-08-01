from marshmallow import Schema, fields, ValidationError, post_load, validate, EXCLUDE
from werkzeug.datastructures import FileStorage
from logging_config import logger

class FileField(fields.Field):
    """Custom field for file uploads."""
    
    def _deserialize(self, value, attr, data, **kwargs):
        if value is None or not isinstance(value, FileStorage):
            raise ValidationError("A valid file is required.")
        return value


class RangeField(fields.Integer):
    """
    Field that maps values from one range to another.
    
    Args:
        input_range: Tuple of (min, max) for input range (default: 0-100)
        output_range: Tuple of (min, max) for output range (default: 0.0-1.0)
        clip: Whether to clip values to the input range (default: True)
        **kwargs: Additional arguments to pass to Float field
    """
    
    def __init__(self, input_range=(0, 100), output_range=(0.0, 1.0), clip=True, is_input_int=True, **kwargs):
        self.input_min, self.input_max = input_range
        self.output_min, self.output_max = output_range
        self.clip = clip
        self.is_input_int = is_input_int

        if is_input_int:
            self.input_min = int(self.input_min)
            self.input_max = int(self.input_max)

        # Handle metadata properly for Marshmallow 4.0+
        metadata = kwargs.pop('metadata', {})
        if 'description' in kwargs:
            metadata['description'] = kwargs.pop('description')
        if metadata:
            kwargs['metadata'] = metadata
        
        # Set up validation for input range
        if 'validate' not in kwargs:
            kwargs['validate'] = validate.Range(
                min=self.input_min,
                max=self.input_max,
                error="Value must be between {min} and {max}, but is {input}"
            )
            
        super().__init__(**kwargs)
    
    def _deserialize(self, value, attr, data, **kwargs):
        try:
            if self.is_input_int:
                value = int(value)
            else:
                value = float(value)

            # Map from input range to output range
            input_span = self.input_max - self.input_min
            output_span = self.output_max - self.output_min
            
            # Avoid division by zero
            if input_span == 0:
                return self.output_min

            # Map the value
            scaled_value = (((value - self.input_min) / input_span) * output_span) - self.output_min

            # Clip to output range
            if self.clip:
                scaled_value = max(self.output_min, min(self.output_max, value))

            return scaled_value
            
        except (TypeError, ValueError) as error:
            raise ValidationError(f"Invalid value ({value}): {error}") from error


class ImageConversionSchema(Schema):
    """Schema for image conversion parameters."""
    class Meta:
        unknown = EXCLUDE  # Ignore extra fields
    
    # File upload
    image = FileField(required=True, metadata={
        'description': 'Image file to convert (PNG, JPG, JPEG)'
    })
    
    # Character dimensions
    char_cols = fields.Integer(
        validate=validate.Range(min=8, max=256, error="Must be between 8 and 256"),
        required=True,
        metadata={'description': 'Number of character columns (8-256)'}
    )
    
    char_rows = fields.Integer(
        validate=validate.Range(min=4, max=256, error="Must be between 4 and 256"),
        required=True,
        metadata={'description': 'Number of character rows (4-256)'}
    )
    
    # Slider values with range mapping
    brightness = RangeField(
        input_range=(0, 100),  # Input from 0-100
        output_range=(-1.0, 1.0),  # Map to -1.0 to 1.0 range
        load_default=0,  # Default value for deserialization
        metadata={'description': 'Brightness adjustment (-1.0 to 1.0)'}
    )

    brightness = fields.Integer(
        validate=validate.Range(min=0, max=100, error="Brightness must be between 0 and 100"),
        load_default=0,  # Default value for deserialization
        metadata={'description': 'Brightness adjustment (0-100)'}
    )
    
    contrast = RangeField(
        validate=validate.Range(min=0, max=100, error="Contrast must be between 0 and 100"),
        output_range=(0.5, 3.0),  # Map to 0.5x to 2.0x
        load_default=1.0,  # Default value for deserialization
        metadata={'description': 'Contrast adjustment (0.5x to 2.0x)'},
    )

def validate_conversion_data(data, files):
    """Validate conversion data against the schema."""
    # Combine form data and files
    form_data = {**data}

    # form data for debugging
    logger.info(f"form_data: {form_data}")

    if 'image' in files:
        form_data['image'] = files['image']
    
    # Validate and deserialize
    schema = ImageConversionSchema()
    try:
        return schema.load(form_data)
    except ValidationError as err:
        return {'errors': err.messages}, 400
