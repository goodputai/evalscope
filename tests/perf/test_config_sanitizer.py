import json
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.config_sanitizer import sanitize_config, sanitize_text


class TestPerfConfigSanitizer(unittest.TestCase):

    def test_arguments_safe_dict_redacts_nested_credentials(self) -> None:
        sentinel = 'secret-sentinel-value'
        args = Arguments(
            model='model',
            url='https://example.com/v1/chat/completions',
            api_key=sentinel,
            headers={
                'Authorization': f'Bearer {sentinel}',
                'X-Api-Key': sentinel,
                'nested': {
                    'proxy-authorization': f'Bearer {sentinel}',
                    'safe': 'visible',
                },
            },
            wandb_api_key=sentinel,
            swanlab_api_key=sentinel,
        )

        safe = args.to_safe_dict()
        serialized = json.dumps(safe, sort_keys=True)

        self.assertNotIn(sentinel, serialized)
        self.assertEqual(safe['api_key'], '<redacted>')
        self.assertEqual(safe['headers']['Authorization'], '<redacted>')
        self.assertEqual(safe['headers']['X-Api-Key'], '<redacted>')
        self.assertEqual(safe['headers']['nested']['safe'], 'visible')
        self.assertNotIn(sentinel, str(args))

    def test_sanitize_config_preserves_non_secret_tokenizer_fields(self) -> None:
        safe = sanitize_config({
            'tokenizer_path': 'ZhipuAI/GLM-5.2',
            'max_tokens': 2048,
            'security_token': 'secret',
        })
        self.assertEqual(safe['tokenizer_path'], 'ZhipuAI/GLM-5.2')
        self.assertEqual(safe['max_tokens'], 2048)
        self.assertEqual(safe['security_token'], '<redacted>')

    def test_sanitize_text_redacts_bearer_assignment_and_signed_url(self) -> None:
        sentinel = 'secret-sentinel-value'
        text = (
            f'Authorization=Bearer {sentinel} '
            f'https://example.com/object?X-Amz-Signature={sentinel}&part=1'
        )
        safe = sanitize_text(text)
        self.assertNotIn(sentinel, safe)
        self.assertIn('part=1', safe)
        self.assertIn('%3Credacted%3E', safe)


if __name__ == '__main__':
    unittest.main()
