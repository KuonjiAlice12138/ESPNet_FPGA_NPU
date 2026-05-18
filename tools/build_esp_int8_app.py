#!/usr/bin/env python3
import vitis


def main():
    client = vitis.create_client()
    client.set_workspace(path=r"D:\ESP_INT8")
    app = client.get_component(name="ESP_INT8_app")
    status = app.build()
    print(f"ESP_INT8_app build status: {status}")
    vitis.dispose()


if __name__ == "__main__":
    main()
